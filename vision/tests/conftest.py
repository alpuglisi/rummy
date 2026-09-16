import sys
from pathlib import Path

# make `import vision` work when pytest is launched from anywhere
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config):
    # register the marker used by the spec ("mark anything > 30 s with @pytest.mark.slow, skip via -m 'not slow'")
    config.addinivalue_line("markers", "slow: tests that run real (tiny) trainings / take longer than ~30 s")
