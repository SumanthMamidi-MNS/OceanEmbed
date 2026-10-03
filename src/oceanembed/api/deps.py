"""Shared FastAPI dependencies and response helpers."""

from __future__ import annotations

import hashlib
from typing import Any

from fastapi import Request
from starlette.responses import Response

from oceanembed.api.encode import NaNSafeJSONResponse
from oceanembed.api.store import ApiError, Run, Store

# Data of a finished run only changes when a pipeline command rewrites run_meta.json, which changes the
# ETag; so the browser may reuse a response for a minute without asking, and revalidates (cheap 304,
# no data read) afterwards.
CACHE_CONTROL = "private, max-age=60, must-revalidate"
EXPOSED_HEADERS = [
    "ETag",
    "X-Shape",
    "X-Dtype",
    "X-Byte-Order",
    "X-Kind",
    "X-Method",
    "X-Date",
    "X-Depth-Index",
    "X-Color-Range",
    "X-Color-Diverging",
    "X-Color-Range-Per-Depth",
    "X-Has-Target",
    "Content-Disposition",
]


class NotModified(Exception):
    def __init__(self, etag: str):
        self.etag = etag


def get_store(request: Request) -> Store:
    return request.app.state.store


def make_etag(run: Run, request: Request) -> str:
    key = "|".join(
        [
            run.name,
            str(run.stamp),
            request.url.path,
            "&".join(sorted(request.url.query.split("&"))),
            request.headers.get("accept", ""),
        ]
    )
    return 'W/"' + hashlib.md5(key.encode("utf-8"), usedforsecurity=False).hexdigest()[:20] + '"'


def run_dep(request: Request, run: str) -> Run:
    """Resolve and validate the ``{run}`` path parameter; answer 304 before any data is read."""
    r = request.app.state.store.run(run)
    etag = make_etag(r, request)
    request.state.etag = etag
    inm = request.headers.get("if-none-match")
    if inm and etag in {t.strip() for t in inm.split(",")}:
        raise NotModified(etag)
    return r


def reply(request: Request, content: Any, *, headers: dict[str, str] | None = None):
    """JSON response (NaN -> null) carrying the run ETag and cache headers."""
    h = dict(headers or {})
    etag = getattr(request.state, "etag", None)
    h["Cache-Control"] = CACHE_CONTROL if etag else "no-store"
    if etag:
        h["ETag"] = etag
    return NaNSafeJSONResponse(content, headers=h)


def reply_bytes(request: Request, body: bytes, media_type: str, headers: dict[str, str]):
    h = dict(headers)
    etag = getattr(request.state, "etag", None)
    h["Cache-Control"] = CACHE_CONTROL
    if etag:
        h["ETag"] = etag
    return Response(content=body, media_type=media_type, headers=h)


def file_headers(request: Request) -> dict[str, str]:
    etag = getattr(request.state, "etag", None)
    h = {"Cache-Control": CACHE_CONTROL}
    if etag:
        h["ETag"] = etag
    return h


def require(cond: bool, status: int, detail: str) -> None:
    if not cond:
        raise ApiError(status, detail)
