import os

import pytest


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("CYPREDICT_RUN_SLOW_IMPORT") != "1",
    reason="Full cyPredict import is slow; set CYPREDICT_RUN_SLOW_IMPORT=1 to run.",
)
def test_full_cypredict_import_with_native_fallback():
    import cyPredict

    assert hasattr(cyPredict, "cyPredict")

