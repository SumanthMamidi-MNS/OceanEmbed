"""Thin re-export of :mod:`oceanembed.data_access`.

The read-only data layer now lives in the package so the HTTP API and the Streamlit dashboard share
one implementation.
"""

from oceanembed.data_access import *  # noqa: F403
from oceanembed.data_access import __all__, _paths  # noqa: F401
