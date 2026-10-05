import pytest

from monolith.runtime import is_available


def pytest_collection_modifyitems(config, items):
    if is_available():
        return
    skip = pytest.mark.skip(reason="monolith.runtime._native is not built (needs macOS + `cmake --build build`)")
    for item in items:
        if "tests/kernels" in str(item.fspath).replace("\\", "/"):
            item.add_marker(skip)
