"""Put the implementation under test on sys.path.

The directory is supplied by EVAL_SRC_DIR so the same reference suite can score
any retained model output (results/<cell>/EVAL-00N/src) or the live eval-project.
This suite lives outside eval-project/ so no model under test can read or edit it.
"""

import os
import sys
import pathlib
import pytest


def pytest_configure(config):
    src = os.environ.get("EVAL_SRC_DIR")
    if not src:
        raise pytest.UsageError("EVAL_SRC_DIR must point at the src/ dir under test")
    path = pathlib.Path(src).resolve()
    if not path.is_dir():
        raise pytest.UsageError(f"EVAL_SRC_DIR is not a directory: {path}")
    sys.path.insert(0, str(path))
