
"""
Natural language parser for constraint-based house planning requests.

This module converts free-form Italian sentences into the JSON structure
understood by the onion algorithm. It relies on an LLM (served via Ollama)
but provides a minimal deterministic fallback for very simple phrases.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Dict, Iterable, List, Optional, Tuple

import config

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - requests is part of requirements
    requests = None

# --------------------------------------------------------------------------- #
# Global parameters
# --------------------------------------------------------------------------- #
DEFAULT_TOLERANCE = 0.10
DEFAULT_NUM_SOLUTIONS = 1
MAX_NUM_SOLUTIONS = 5
LLM_MAX_RETRIES = 2
OLLAMA_TIMEOUT_SECONDS = 120

LOGGER = logging.getLogger(__name__)

_REQUESTS_SESSION = None

SYSTEM_PROMPT = dedent(
    """
    Sei un traduttore che converte richieste in linguaggio naturale (italiano)
    nel formato JSON atteso dall'algoritmo onion. Segui SEMPRE queste regole:

    - Rispondi con un UNICO oggetto JSON valido, senza testo extra.
    - Struttura obbligatoria:
      {{
        "type": "onion_algorithm",
        "goals": [[OPERATOR, ["area", "ROOM_ID"], VALORE]],
        "preserve": [[OPERATOR, ["area", "ROOM_ID"], VALORE_O_STRING]],
        "num_solutions": intero,
        "tolerance": numero_decimale
      }}
    - OPERATOR è uno tra ">", ">=", "<", "<=", "==".
    - Usa ESATTAMENTE gli ID stanza (es. s#room_5#RIPOSTIGLIO) forniti nel mapping contestuale.
    - Se la richiesta chiede di mantenere l'area originale di una stanza, usa
      il valore numerico esatto dell'area iniziale (campo area nel mapping) come numero (es. 51.12).
    - Analizza ogni frase della richiesta: per ciascun vincolo numerico (maggiore di, almeno, non più di, ecc.)
      inserisci un constraint. Il primo vincolo cronologicamente diventa il goal, tutti gli altri finiscono in preserve.
    - Quando l'utente elenca più stanze dopo espressioni come "mantenendo l'area del bagno, cucina e disimpegno",
      crea un preserve separato per OGNI stanza menzionata usando l'ID corretto e il valore dell'area iniziale corrispondente.
    - Non omettere MAI una stanza o un vincolo esplicitamente menzionato (anche se congiunto da "e" oppure separato da virgole).
    - Se l'utente non specifica il numero di soluzioni, imposta num_solutions a {default_num_solutions}.
    - Se l'utente non specifica la tolleranza, imposta tolerance a {default_tolerance}.
    - Usa SEMPRE i doppi apici (\") per chiavi e stringhe.
    - Non usare segnaposto generici tipo "NOME STANZA": sostituiscili con uno dei nomi indicati.
    - Non inventare stanze: usa solo i nomi forniti.
    - Se la richiesta contiene più vincoli sulla stessa stanza, riportali tutti.
    """
).format(
    default_num_solutions=DEFAULT_NUM_SOLUTIONS,
    default_tolerance=DEFAULT_TOLERANCE,
).strip()

PROMPT_TEMPLATE = dedent(
    """
    CONTEXT:
    Stanze disponibili (usa il nome esatto nel placeholder <ROOM_ID:...>):
    {room_catalog}

    Esempio di output:
    {{
      "type": "onion_algorithm",
      "goals": [[">", ["area", "s#room_5#RIPOSTIGLIO"], <user_requested_area>]],
      "preserve": [
        ["==", ["area", "s#room_4#DISIMPEGNO"], <initial_area_of_DISIMPEGNO>],
        ["==", ["area", "s#room_7#BAGNO"], <initial_area_of_BAGNO>],
        ["==", ["area", "s#room_2#CUCINA"], <initial_area_of_CUCINA>]
      ],
      "num_solutions": <user_requested_num_solutions>,
      "tolerance": <user_requested_tolerance>
    }}

    Format Guidance:
    - Ricorda: restituisci SOLO l'oggetto JSON senza testo extra.
    - Valori numerici in metri quadrati devono essere numeri (non stringhe) con punto come separatore decimale.
    - Per ogni stanza citata usa l'ID e l'area iniziale corrispondenti dal mapping (nessun placeholder aggiuntivo).

    Richiesta utente:
    {user_request}
    """
).strip()


@dataclass
class ParsedConstraint:
    operator: str
    room_id: str
    numeric_value: float
    position: int = 0


class NLParser:
    """Convert Italian natural language requests into structured constraints."""

    def __init__(self, room_mapper):
        self.room_mapper = room_mapper
        self.last_llm_json: Optional[str] = None

    def parse_request(self, user_request: str) -> Dict[str, Any]:
        if not user_request or not user_request.strip():
            raise ValueError("La richiesta è vuota.")

        resolved_request = self._resolve_room_labels(user_request)
        if resolved_request != user_request:
            LOGGER.debug("Resolved room labels in request: %s", resolved_request)

        try:
            llm_response = self._query_llm(resolved_request)
            parsed, llm_json = self._convert_llm_output(llm_response)
            self.last_llm_json = llm_json
            print(f"\n LLM response: {self.last_llm_json}")
            LOGGER.debug("LLM parse succeeded.")
            return parsed
        except Exception as exc:
            LOGGER.warning("LLM parse failed (%s), falling back to deterministic parser.", exc)
            self.last_llm_json = None
            print("\n LLM response: fallback deterministic parser (nessun output LLM)")
            return self._deterministic_parse(user_request)

    def _query_llm(self, user_request: str) -> str:
        if requests is None:
            raise RuntimeError(
                "La libreria 'requests' non è disponibile: impossibile interrogare l'LLM."
            )

        room_catalog = self._build_room_catalog()
        prompt = PROMPT_TEMPLATE.format(
            room_catalog=room_catalog,
            user_request=user_request.strip(),
        )

        session = self._ensure_requests_session()
        url = f"http://{config.OLLAMA_HOST}/api/generate"
        payload = {
            "model": config.OLLAMA_MODEL,
            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
            "format": "json",
        }

        response = session.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
        response.raise_for_status()

        text = response.text.strip()
        if not text:
            raise ValueError("Risposta vuota dall'LLM.")

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "response" in parsed:
                response_content = parsed.get("response", "")
                if isinstance(response_content, dict):
                    return json.dumps(response_content)
                return response_content
        except json.JSONDecodeError:
            pass

        accumulated: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                response_content = chunk.get("response", "")
                if isinstance(response_content, dict):
                    accumulated.append(json.dumps(response_content))
                else:
                    accumulated.append(response_content)
            except json.JSONDecodeError:
                accumulated.append(line)
        return "".join(accumulated)

    def _ensure_requests_session(self):
        global _REQUESTS_SESSION
        if _REQUESTS_SESSION is None:
            _REQUESTS_SESSION = requests.Session()
        return _REQUESTS_SESSION

    def _resolve_room_labels(self, user_request: str) -> str:
        if not user_request or not hasattr(self.room_mapper, "get_label_resolution_patterns"):
            return user_request

        try:
            patterns = self.room_mapper.get_label_resolution_patterns()
        except Exception as exc:
            LOGGER.debug("Unable to build label resolution patterns: %s", exc)
            return user_request

        resolved_request = user_request
        for pattern, room_id in patterns:
            def _replacement(match: re.Match) -> str:
                span_start = match.start()
                candidate = match.group(0)
                if "s#room" in candidate.lower():
                    return candidate
                prefix = match.string[max(0, span_start - 7):span_start].lower()
                if "s#room" in prefix:
                    return candidate
                return room_id

            resolved_request = pattern.sub(_replacement, resolved_request)

        return resolved_request

    def _build_room_catalog(self) -> str:
        rooms = []
        try:
            for room in self.room_mapper.list_rooms():
                rooms.append(
                    f"- {room['name']} -> <ROOM_ID:{room['id']}> (area: {room['area']:.2f} m²)"
                )
        except Exception as exc:
            LOGGER.debug("Unable to build room catalog: %s", exc)
        return "\n".join(rooms)

    def _convert_llm_output(self, text: str) -> Tuple[Dict[str, Any], str]:
        if not text:
            raise ValueError("Risposta vuota dall'LLM.")

        data: Optional[Dict[str, Any]] = None
        json_str: Optional[str] = None

        try:
            maybe_json = json.loads(text)
            if isinstance(maybe_json, dict):
                data = maybe_json
                json_str = json.dumps(maybe_json)
        except json.JSONDecodeError:
            pass

        if data is None:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ValueError(f"Nessun JSON trovato nella risposta dell'LLM: {text}")
            json_str = match.group(0)
            data = json.loads(json_str)

        if not isinstance(data, dict):
            raise ValueError("Il risultato dell'LLM non è un oggetto JSON.")

        data.setdefault("type", "onion_algorithm")
        if data["type"] != "onion_algorithm":
            raise ValueError("Il JSON dell'LLM deve avere type='onion_algorithm'.")

        data.setdefault("goals", [])
        data.setdefault("preserve", [])
        data.setdefault("num_solutions", DEFAULT_NUM_SOLUTIONS)
        data.setdefault("tolerance", DEFAULT_TOLERANCE)

        data["goals"] = self._normalise_constraints(data["goals"], "goal")
        data["preserve"] = self._normalise_constraints(data["preserve"], "preserve")

        try:
            num_solutions = int(data.get("num_solutions", DEFAULT_NUM_SOLUTIONS))
            data["num_solutions"] = max(1, min(MAX_NUM_SOLUTIONS, num_solutions))
        except Exception:
            data["num_solutions"] = DEFAULT_NUM_SOLUTIONS

        try:
            tolerance = float(str(data.get("tolerance", DEFAULT_TOLERANCE)).replace(",", "."))
            data["tolerance"] = max(0.0, tolerance)
        except Exception:
            data["tolerance"] = DEFAULT_TOLERANCE

        return data, json_str or json.dumps(data)

    def _normalise_constraints(self, constraints: Iterable[Any], kind: str) -> List[List[Any]]:
        normalised: List[List[Any]] = []
        if not constraints:
            return normalised

        for constraint in constraints:
            if (
                not isinstance(constraint, (list, tuple))
                or len(constraint) < 3
                or not isinstance(constraint[1], (list, tuple))
            ):
                raise ValueError(f"Constraint {kind} malformato: {constraint}")

            operator = str(constraint[0]).strip()
            area_clause = list(constraint[1])
            if len(area_clause) != 2 or area_clause[0] != "area":
                raise ValueError(f"Clause area malformata: {constraint}")

            room_id = area_clause[1]
            if not self._is_valid_room_id(room_id):
                raise ValueError(f"ROOM_ID sconosciuto o non valido: {room_id}")

            value = constraint[2]
            if isinstance(value, str):
                value_clean = value.strip().upper()
                if value_clean == "INITIAL_AREA":
                    value = "INITIAL_AREA"
                else:
                    value = float(value.replace(",", "."))
            elif isinstance(value, (int, float)):
                value = float(value)
            else:
                raise ValueError(f"Valore target non valido: {value}")

            normalised.append([operator, ["area", room_id], value])

        return normalised

    def _is_valid_room_id(self, room_id: str) -> bool:
        try:
            return bool(self.room_mapper.room_map.get(room_id))
        except Exception:
            return False

    def _deterministic_parse(self, user_request: str) -> Dict[str, Any]:
        text = self._normalise_text(user_request)
        pattern = re.compile(
            r"(?:il|lo|la|l'|al|alla|allo|nel|nello|nella|del|dello|della)?\s*"
            r"([a-zàèéìòù_]+)\s*(\d+)?\s*(?:#\s*)?(\d+)?\s*(?:maggiore di|almeno|non meno di|superiore a)\s*(\d+(?:[\.,]\d+)?)",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if not match:
            raise ValueError("Impossibile interpretare la richiesta senza LLM.")

        base = match.group(1)
        explicit_number = match.group(2) or match.group(3)
        value = float(match.group(4).replace(",", "."))

        room_query = base
        if explicit_number:
            room_query += f" {explicit_number}"

        room_id = self.room_mapper.find_room_id(room_query)
        if not room_id:
            raise ValueError(f"Impossibile mappare la stanza '{room_query}'.")

        goals = [[">", ["area", room_id], value]]
        preserve: List[List[Any]] = []

        return {
            "type": "onion_algorithm",
            "goals": goals,
            "preserve": preserve,
            "num_solutions": DEFAULT_NUM_SOLUTIONS,
            "tolerance": DEFAULT_TOLERANCE,
        }

    @staticmethod
    def _normalise_text(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = text.replace("m2", "m²").replace("mq", "m²")
        return text
