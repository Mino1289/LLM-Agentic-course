from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any


async def sse_event_stream(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
    yield ": stream-open\n\n"
    async for event in events:
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
