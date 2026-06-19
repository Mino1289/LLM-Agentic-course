from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.paths import REPORTS_DIR

router = APIRouter(prefix="/api", tags=["reports"])


@router.get("/reports/{filename}")
async def download_report(filename: str) -> FileResponse:
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = (REPORTS_DIR / filename).resolve()
    if not str(path).startswith(str(REPORTS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")

    if not path.is_file():
        raise HTTPException(status_code=404, detail="Report not found")

    media_type = "application/pdf" if path.suffix.lower() == ".pdf" else "text/markdown"
    return FileResponse(path, filename=path.name, media_type=media_type)
