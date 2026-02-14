import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
