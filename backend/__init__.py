import sys
from pathlib import Path

_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.append(_root_dir)

_ai_service_dir = str(Path(__file__).resolve().parent.parent / "ai-service")
if _ai_service_dir not in sys.path:
    sys.path.append(_ai_service_dir)
