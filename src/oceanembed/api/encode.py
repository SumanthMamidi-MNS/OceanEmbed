"""JSON encoding for the API: NaN / inf become ``null``, numpy arrays become nested lists.

Starlette's ``JSONResponse`` refuses NaN (``allow_nan=False``), and plain ``json.dumps`` would write
the invalid token ``NaN``. :func:`clean` walks a payload once and converts everything to strictly valid
JSON types; :class:`NaNSafeJSONResponse` is the default response class of the app.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from typing import Any

import numpy as np
import pandas as pd
from starlette.responses import JSONResponse

DEFAULT_DECIMALS = 3


def array_to_list(a: np.ndarray, decimals: int | None = DEFAULT_DECIMALS):
    """Nested list of Python floats / ints / ``None`` (NaN and inf -> ``None``).

    Floats are rounded in float64 first (rounding a float32 and printing it would give
    ``28.123000144958496``); integers and booleans pass through unchanged.
    """
    a = np.asarray(a)
    if a.dtype.kind == "b":
        return a.tolist()
    if a.dtype.kind in "iu":
        return a.tolist()
    if a.dtype.kind == "M":
        return [None if pd.isna(v) else pd.Timestamp(v).isoformat() for v in a.ravel()]
    a = a.astype(np.float64, copy=False)
    bad = ~np.isfinite(a)
    if decimals is not None:
        a = np.round(a, decimals)
    if not bad.any():
        return a.tolist()
    o = a.astype(object)
    o[bad] = None
    return o.tolist()


def clean(obj: Any, decimals: int | None = None) -> Any:
    """Convert ``obj`` to strictly valid JSON types (recursively)."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if not math.isfinite(f) else (f if decimals is None else round(f, decimals))
    if isinstance(obj, np.ndarray):
        return array_to_list(obj, DEFAULT_DECIMALS if decimals is None else decimals)
    if isinstance(obj, dict):
        return {str(k): clean(v, decimals) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v, decimals) for v in obj]
    if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date, np.datetime64)):
        return None if pd.isna(obj) else pd.Timestamp(obj).isoformat()
    if isinstance(obj, pd.DatetimeIndex):
        return [d.isoformat() for d in obj]
    if isinstance(obj, (pd.Series, pd.Index)):
        return clean(obj.to_numpy(), decimals)
    if hasattr(obj, "model_dump"):
        return clean(obj.model_dump(mode="json"), decimals)
    raise TypeError(f"cannot encode {type(obj).__name__} as JSON")


class NaNSafeJSONResponse(JSONResponse):
    """JSON response that writes NaN / inf as ``null`` and numpy arrays as lists."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            clean(content), ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
