import pytest

from src.data import binarise, dedupe
from src.ingest import load_raw_file


@pytest.fixture(scope="session")
def raw():
    return load_raw_file()


@pytest.fixture(scope="session")
def clean(raw):
    return dedupe(binarise(raw))[0]
