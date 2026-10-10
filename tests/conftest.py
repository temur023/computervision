import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SWARMSIGHT_WARM", "0")
os.environ.setdefault("SWARMSIGHT_RUNTIME", tempfile.mkdtemp(prefix="swarmsight_test_"))
