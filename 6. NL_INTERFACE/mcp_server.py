#!/usr/bin/env python3
"""
MCP Server for Natural Language House Planner
Model Context Protocol server that exposes house planning operations
"""
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

# Add house-planner to path
sys.path.insert(0, str(Path(__file__).parent.parent / "5. ENGINE" / "house-planner" / "src"))

from houseplanner.io.parser import load_house
from room_mapper import RoomMapper
from nl_parser import NLParser
from executor import Executor
import config


class HousePlannerMCP:
    """MCP Server for House Planner"""
    
    def __init__(self):
        self.house_path = str(Path(__file__).parent / config.HOUSE_DATA_PATH)
        self.house = load_house(self.house_path)
        self.mapper = RoomMapper(self.house, house_path=self.house_path)
        self.parser = NLParser(self.mapper)
        self.executor = Executor(self.house_path)
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle MCP request
        
        Request format:
        {
            "method": "plan_house",
            "params": {
                "request": "Natural language request"
            }
        }
        
        Response format:
        {
            "result": {
                "success": bool,
                "type": "constraint" | "operation",
                "output_dir": str,
                "solutions": [...] or operation result
            }
        }
        """
        try:
            method = request.get('method')
            params = request.get('params', {})
            
            if method == 'plan_house':
                return await self._plan_house(params)
            elif method == 'list_rooms':
                return await self._list_rooms()
            elif method == 'get_capabilities':
                return await self._get_capabilities()
            else:
                return {
                    'error': {
                        'code': -32601,
                        'message': f'Method not found: {method}'
                    }
                }
                
        except Exception as e:
            return {
                'error': {
                    'code': -32603,
                    'message': f'Internal error: {str(e)}'
                }
            }
    
    async def _plan_house(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute house planning request"""
        user_request = params.get('request')
        if not user_request:
            return {
                'error': {
                    'code': -32602,
                    'message': 'Missing required parameter: request'
                }
            }
        
        try:
            # Parse request
            parsed = self.parser.parse_request(user_request)
            
            # Execute request
            result = self.executor.execute(parsed, user_request)
            
            # Format response
            response = {
                'result': {
                    'success': result.get('success', False),
                    'type': result.get('type'),
                    'output_dir': result.get('output_dir'),
                }
            }
            
            if result.get('type') == 'constraint':
                response['result']['solutions_found'] = result.get('solutions_found', 0)
                response['result']['execution_time'] = result.get('execution_time', 0)
                if result.get('solutions'):
                    response['result']['solutions'] = [
                        {
                            'index': sol['index'],
                            'image_path': sol['image_path'],
                            'operations_count': len(sol['operations']),
                            'success': sol['success']
                        }
                        for sol in result['solutions']
                    ]
            elif result.get('type') == 'operation':
                response['result']['execution_time'] = result.get('execution_time', 0)
                response['result']['doors_changed'] = {
                    'closed': result.get('doors_closed', []),
                    'opened': result.get('doors_opened', [])
                }
                response['result']['final_image'] = result.get('final_image')
            
            if not result.get('success'):
                response['result']['error'] = result.get('error')
            
            return response
            
        except Exception as e:
            return {
                'error': {
                    'code': -32603,
                    'message': f'Execution failed: {str(e)}'
                }
            }
    
    async def _list_rooms(self) -> Dict[str, Any]:
        """List all available rooms"""
        rooms = self.mapper.list_rooms()
        return {
            'result': {
                'rooms': [
                    {
                        'name': room['name'],
                        'number': room['number'],
                        'id': room['id'],
                        'area': room['area']
                    }
                    for room in rooms
                ]
            }
        }
    
    async def _get_capabilities(self) -> Dict[str, Any]:
        """Get server capabilities"""
        return {
            'result': {
                'capabilities': {
                    'methods': [
                        {
                            'name': 'plan_house',
                            'description': 'Execute natural language house planning request',
                            'params': {
                                'request': 'Natural language request in Italian'
                            }
                        },
                        {
                            'name': 'list_rooms',
                            'description': 'List all available rooms in the house',
                            'params': {}
                        },
                        {
                            'name': 'get_capabilities',
                            'description': 'Get server capabilities',
                            'params': {}
                        }
                    ],
                    'examples': [
                        {
                            'request': 'Voglio che il bagno sia maggiore di 26 m²',
                            'description': 'Simple constraint to expand bathroom'
                        },
                        {
                            'request': 'Voglio che il bagno sia maggiore di 26 m² mantenendo il disimpegno',
                            'description': 'Constraint with preservation'
                        },
                        {
                            'request': 'Vorrei chiudere la porta sul balcone ed aprirla nello studio',
                            'description': 'Door operation'
                        }
                    ]
                }
            }
        }


async def run_server(host: str = 'localhost', port: int = 8080):
    """Run MCP server"""
    server = HousePlannerMCP()
    
    print(f" House Planner MCP Server")
    print(f"Listening on {host}:{port}")
    print(f"\nAvailable methods:")
    print(f"  - plan_house: Execute natural language request")
    print(f"  - list_rooms: List all rooms")
    print(f"  - get_capabilities: Get server info")
    
    # Simple JSON-RPC server over stdin/stdout
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            request = json.loads(line)
            response = await server.handle_request(request)
            print(json.dumps(response))
            sys.stdout.flush()
            
        except json.JSONDecodeError as e:
            error_response = {
                'error': {
                    'code': -32700,
                    'message': f'Parse error: {str(e)}'
                }
            }
            print(json.dumps(error_response))
            sys.stdout.flush()
        except KeyboardInterrupt:
            break


def test_mcp_server():
    """Test MCP server with sample requests"""
    print("=" * 80)
    print(" Testing MCP Server")
    print("=" * 80)
    
    server = HousePlannerMCP()
    
    # Test 1: Get capabilities
    print("\n1. Testing get_capabilities...")
    request = {
        'method': 'get_capabilities',
        'params': {}
    }
    
    async def run_test():
        response = await server.handle_request(request)
        print(json.dumps(response, indent=2))
    
    asyncio.run(run_test())
    
    # Test 2: List rooms
    print("\n2. Testing list_rooms...")
    request = {
        'method': 'list_rooms',
        'params': {}
    }
    
    async def run_test2():
        response = await server.handle_request(request)
        print(json.dumps(response, indent=2))
    
    asyncio.run(run_test2())
    
    print("\n MCP Server tests completed!")
    print("\nTo test plan_house, run:")
    print('  echo \'{"method":"plan_house","params":{"request":"Voglio che il bagno sia maggiore di 26 m²"}}\' | python mcp_server.py')


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_mcp_server()
    else:
        asyncio.run(run_server())





