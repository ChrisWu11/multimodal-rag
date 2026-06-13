from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import get_settings
from app.core.container import get_container


ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).parent / "static"
FRONTEND_DIST_DIR = ROOT_DIR / "frontend" / "dist"


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description="Multimodal RAG API for ultrasound, thermal imaging, and evidence-grounded Q&A.",
        version="0.1.0",
    )

    get_container()
    app.include_router(router, prefix=settings.api_prefix)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    frontend_assets = FRONTEND_DIST_DIR / "assets"
    if frontend_assets.exists():
        app.mount("/assets", StaticFiles(directory=frontend_assets), name="assets")

    @app.get("/", response_class=HTMLResponse)
    def chat_ui() -> str:
        index_path = FRONTEND_DIST_DIR / "index.html"
        if index_path.exists():
            return index_path.read_text(encoding="utf-8")
        return _missing_frontend_html()

    @app.get("/debug", response_class=HTMLResponse)
    def debug_ui() -> str:
        return (STATIC_DIR / "debug.html").read_text(encoding="utf-8")

    return app


app = create_app()


def _missing_frontend_html() -> str:
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Frontend build required</title>
        <style>
          body {
            margin: 0;
            min-height: 100vh;
            display: grid;
            place-items: center;
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f5f6f1;
            color: #202527;
          }
          main {
            max-width: 640px;
            border: 1px solid #dce2da;
            border-radius: 8px;
            background: white;
            padding: 28px;
          }
          code {
            display: block;
            margin-top: 12px;
            padding: 12px;
            border-radius: 6px;
            background: #eff3ef;
          }
          a { color: #0d7c76; font-weight: 700; }
        </style>
      </head>
      <body>
        <main>
          <h1>Frontend build required</h1>
          <p>Build the React demo UI before opening the main chat page.</p>
          <code>cd frontend && npm install && npm run build</code>
          <p><a href="/debug">Open debug UI</a></p>
        </main>
      </body>
    </html>
    """
