"""Put ``src/`` on the path for the standalone scripts."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
