import sqlite3

import pytest

from sentinel.config import get_settings
from sentinel.db import bootstrap


@pytest.fixture()
def conn(tmp_path):
    """Fresh in-memory store bootstrapped from the repo dataset."""
    settings = get_settings()
    # check_same_thread=False: FastAPI's TestClient serves requests from a
    # worker thread but the fixture connection is created on the test thread.
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    bootstrap(connection, data_dir=settings.data_dir)
    yield connection
    connection.close()
