"""Make the `cdk/` directory importable the same way `cdk.json` runs it.

`cdk.json` runs `python3 cdk/app.py`, which puts `cdk/` on `sys.path`, so modules
there import each other by bare name (`from settings import StackSettings`).
"""

import sys
from pathlib import Path

CDK_DIR = Path(__file__).parents[2] / "cdk"

if str(CDK_DIR) not in sys.path:
    sys.path.insert(0, str(CDK_DIR))
