"""VLM-first card-oracle pipeline for shoebuyer catalog extraction.

See README.md for architecture and design rationale.
"""
from dotenv import load_dotenv as _load_dotenv

# Load ANTHROPIC_API_KEY (and any other env vars) from repo-root .env at import.
# Walks parents from this file's location until it finds a .env, so the package
# works whether invoked from the repo root or from inside buysheet_v2/.
_load_dotenv()

__version__ = "0.1.0"
