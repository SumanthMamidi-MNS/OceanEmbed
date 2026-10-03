"""Read-only HTTP data API (FastAPI) for the OceanEmbed dashboard: ``oceanembed serve``."""

from oceanembed.api.app import create_app

__all__ = ["create_app"]
