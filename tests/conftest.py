"""Every tier: a Metal shader-validation report fails the test phase (setup, call or teardown) in which a command buffer
carried it. The runtime raises each report as ``RuntimeError: shader validation: …``; this catches the ones that never
reach pytest: an ``except Exception`` in a harness, or a raise on another thread, which pytest only turns into a warning.
"""
import pytest

from monolith.runtime import _native

_reports = getattr(_native, "validation_reports", lambda: 0)   # 0 without the Metal module


def _checked(when):
    before = _reports()
    result = yield                          # a phase that raised re-raises here and fails as it is
    if n := _reports() - before:
        pytest.fail(f"{n} Metal shader-validation report(s) in this {when} phase that nothing raised (an exception caught, "
                    "or raised on another thread); the reports are on stderr with MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1, "
                    "else in the unified log", pytrace=False)
    return result


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    return (yield from _checked("setup"))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    return (yield from _checked("call"))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item):
    return (yield from _checked("teardown"))
