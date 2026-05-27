from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import get_settings
from app.core.container import get_container


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description="Multimodal RAG API for ultrasound, thermal imaging, and evidence-grounded Q&A.",
        version="0.1.0",
    )

    get_container()
    app.include_router(router, prefix=settings.api_prefix)

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    def debug_ui() -> str:
        return (static_dir / "index.html").read_text(encoding="utf-8")

    return app


app = create_app()
