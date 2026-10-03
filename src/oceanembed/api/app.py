"""FastAPI application factory.

``create_app(outputs_root, web_dist)`` builds the app: JSON API under ``/api`` (OpenAPI page at
``/api/docs``), CORS only for the Vite dev server, GZip, clean JSON errors, and -- when the front end has
been built (``web/dist/index.html``) -- the single-page app at ``/`` with a client-side-route fallback.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import FileResponse, JSONResponse, Response

from oceanembed.api import routes_fields, routes_metrics, routes_runs, schemas
from oceanembed.api.deps import CACHE_CONTROL, EXPOSED_HEADERS, NotModified
from oceanembed.api.encode import NaNSafeJSONResponse
from oceanembed.api.store import ApiError, Store
from oceanembed.config import OUTPUTS_ROOT_ENV

log = logging.getLogger("oceanembed.api")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
API_VERSION = "0.1.0"


def default_outputs_root() -> Path:
    """``OCEANEMBED_OUTPUTS_ROOT`` or ``<project root>/outputs`` (same rule as the Streamlit app)."""
    env = os.environ.get(OUTPUTS_ROOT_ENV)
    return Path(env).resolve() if env else PROJECT_ROOT / "outputs"


def _error(status: int, detail: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status, headers=headers)


def _warm(store: Store) -> None:
    """Open the libraries, the Zarr handles, the error maps and the embedding PCA once, off the request
    path (the first cold use of a run otherwise costs a few seconds)."""
    try:
        from oceanembed import data_access as da

        for name in store.run_names():
            run = store.run(name)
            days = store.dates(run)
            da._open_store(run.path)
            if len(days):
                store.climatology(run, days[0])  # loads the statistics file once
            emb = da.embedding_dates(run.path)
            if len(emb):
                da.embedding_pca_rgb(run.path, emb[0])  # fits (and caches) the run's PCA
            store.map_arrays(run)
            store.field_methods(run)
    except Exception:  # noqa: BLE001 - warming is best effort
        log.debug("warm-up failed", exc_info=True)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    threading.Thread(target=_warm, args=(app.state.store,), daemon=True, name="api-warm").start()
    yield


def create_app(
    outputs_root: str | Path | None = None, web_dist: str | Path | None = None
) -> FastAPI:
    root = Path(outputs_root).resolve() if outputs_root else default_outputs_root()
    dist = Path(web_dist).resolve() if web_dist else PROJECT_ROOT / "web" / "dist"

    app = FastAPI(
        title="OceanEmbed data API",
        version=API_VERSION,
        description=(
            "Read-only JSON / binary API over the finished runs in `outputs/`: subsurface temperature "
            "reconstructed from surface satellite fields (15 depths, 0-1000 m, North Indian Ocean). "
            'NaN is always `null`. Errors are `{"detail": ...}` with 400 (bad parameter) or 404 '
            "(unknown run / missing artefact / no data). See docs/api.md."
        ),
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        default_response_class=NaNSafeJSONResponse,
        lifespan=_lifespan,
    )
    app.state.store = Store(root)
    app.state.web_dist = dist

    app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=4)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_methods=["GET", "HEAD", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=EXPOSED_HEADERS,
    )

    # ------------------------------------------------------------------ errors
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return _error(exc.status, exc.detail)

    @app.exception_handler(NotModified)
    async def _not_modified(_: Request, exc: NotModified):
        return Response(status_code=304, headers={"ETag": exc.etag, "Cache-Control": CACHE_CONTROL})

    @app.exception_handler(RequestValidationError)
    async def _bad_request(_: Request, exc: RequestValidationError):
        parts = []
        for e in exc.errors():
            loc = ".".join(str(p) for p in e.get("loc", ()) if p not in ("query", "path"))
            parts.append(
                f"{loc}: {e.get('msg', 'invalid')}" if loc else str(e.get("msg", "invalid"))
            )
        return _error(400, "invalid parameter - " + "; ".join(parts))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        if exc.status_code in (204, 304):
            return Response(status_code=exc.status_code, headers=exc.headers)
        return _error(exc.status_code, str(exc.detail), exc.headers)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return _error(500, "internal error while reading the run data")

    # ------------------------------------------------------------------ API
    @app.get("/api/health", response_model=schemas.Health, summary="Liveness and basic facts")
    def health(request: Request):
        d: Path = request.app.state.web_dist
        return NaNSafeJSONResponse(
            {
                "status": "ok",
                "version": API_VERSION,
                "n_runs": len(request.app.state.store.run_names()),
                "ui_built": (d / "index.html").is_file(),
            },
            headers={"Cache-Control": "no-store"},
        )

    for module in (routes_runs, routes_fields, routes_metrics):
        app.include_router(module.router, prefix="/api")

    # ------------------------------------------------------------------ single-page app (last)
    @app.get("/{path:path}", include_in_schema=False)
    def spa(request: Request, path: str):
        if path == "api" or path.startswith("api/"):
            return _error(404, f"no such API endpoint: /{path}")
        d: Path = request.app.state.web_dist
        index = d / "index.html"
        if not index.is_file():
            if path == "":
                return NaNSafeJSONResponse(
                    {
                        "message": "The OceanEmbed web UI is not built. The data API is running.",
                        "hint": "cd web && npm install && npm run build  (or run the Vite dev server)",
                        "api_docs": "/api/docs",
                    }
                )
            return _error(404, "not found")
        if path:
            target = (d / path).resolve()
            if target.is_file() and d.resolve() in target.parents:
                cache = (
                    "public, max-age=31536000, immutable"
                    if path.startswith("assets/")
                    else "no-cache"
                )
                return FileResponse(target, headers={"Cache-Control": cache})
            if "." in Path(path).name:  # a missing asset must not turn into the HTML shell
                return _error(404, "not found")
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return app


def create_app_from_env() -> FastAPI:
    """Factory for ``uvicorn --reload`` (settings travel through the environment)."""
    return create_app(os.environ.get("OCEANEMBED_API_OUTPUTS_ROOT") or None)
