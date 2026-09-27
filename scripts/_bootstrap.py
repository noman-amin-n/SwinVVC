"""
Make the `swinvvc` package importable when running scripts from a source
checkout without installing it (``pip install -e .`` makes this unnecessary).
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
