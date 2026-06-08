from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from matchtrace import __version__
from matchtrace.inference.pipeline import analyze_video
from matchtrace.utils.config import load_settings, resolve_path


class AnalyzeRequest(BaseModel):
    input_name: str = Field(min_length=1, max_length=200)
    output_name: str = Field(
        default="api_analysis.mp4",
        min_length=1,
        max_length=200,
    )


def _safe_child(root: Path, name: str) -> Path:
    if Path(name).name != name:
        raise ValueError("Only a file name without directory components is allowed.")
    candidate = (root / name).resolve()
    if candidate.parent != root.resolve():
        raise ValueError("Requested path is outside the configured root.")
    return candidate


def create_app(config_path: str | Path | None = None) -> FastAPI:
    settings = load_settings(config_path)
    input_root = resolve_path(settings, settings["api"]["input_root"])
    output_root = resolve_path(settings, settings["api"]["output_root"])
    app = FastAPI(
        title="MatchTrace CV API",
        version=__version__,
        description="Synchronous football-video analysis API.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "matchtrace-cv"}

    @app.post("/analyze")
    def analyze(request: AnalyzeRequest) -> dict:
        try:
            source = _safe_child(input_root, request.input_name)
            output = _safe_child(output_root, request.output_name)
            summary = output.with_suffix(".json")
            return analyze_video(settings, source, output, summary)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
