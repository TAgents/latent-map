import pytest

from latent_capture.loader import load_model


@pytest.fixture(scope="session")
def loaded():
    return load_model()
