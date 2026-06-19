from __future__ import annotations

from fastapi import APIRouter

from src.tools.definitions import get_tool_definitions

router = APIRouter(prefix="/api", tags=["tools"])


@router.get("/tools")
async def list_tools() -> dict:
    tools = []
    for tool in get_tool_definitions():
        fn = tool.get("function", {})
        tools.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        )
    return {"tools": tools}
