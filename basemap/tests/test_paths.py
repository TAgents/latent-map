from mapbuild.paths import paths


def test_v0_keeps_its_old_file_names_and_later_versions_are_tagged():
    assert paths("v0")["corpus"].name == "corpus.jsonl" and paths("v0")["bundle"].name == "bundle_v0"
    assert paths("v1")["states"].name == "states_v1.npz" and paths("v1")["bundle"].name == "bundle_v1"
