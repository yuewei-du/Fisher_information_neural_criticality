import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# The repo is not pip-installable yet, so make ``fim`` and ``examples``
# importable from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: end-to-end tests that run the full estimator"
    )
