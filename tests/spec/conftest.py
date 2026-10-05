import pytest


def pytest_collection_modifyitems(config, items):
    try:
        import torch  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(reason="torch is not installed (pip install -e '.[oracle]')")
        for item in items:
            if "tests/spec" in str(item.fspath).replace("\\", "/"):
                item.add_marker(skip)
