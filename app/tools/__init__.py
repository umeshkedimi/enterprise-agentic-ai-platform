"""Tool registry package.

`builtin` is imported for its side effect: registration happens at import time,
so a tool module nothing imports is invisible to every allowlist that names it.
Same hazard as a SQLModel table missing from `app/db/base.py` — the failure is
silent and reads like a configuration mistake rather than a missing import, so
new tool modules must be added here.
"""

from app.tools import builtin  # noqa: F401
