"""Room area computation using the same pipeline as areas.py.

This module calls compute_areas function directly without subprocess overhead.
"""

from __future__ import annotations

from typing import Dict

from ..core.model import House
from ..visualization.generator import _convert_house_to_dict
from .areas import compute_areas


def compute_room_areas_with_areas_py(house: House) -> Dict[str, float]:
    """Compute room areas (m^2) using the same logic implemented in areas.py.

    Returns a mapping room_id -> area_m2. Missing rooms are omitted.
    
    Optimized: calls compute_areas function directly instead of subprocess.
    """
    # Convert house to dict format
    house_dict = _convert_house_to_dict(house)
    
    # Call compute_areas in areas-only mode (no I/O, no plotting)
    result = compute_areas(
        house_dict,
        build_png=False,
        build_csv=False,
        build_json=False
    )

    # Extract areas from result
    areas: Dict[str, float] = result.get("areas_by_id", {})

    return areas
