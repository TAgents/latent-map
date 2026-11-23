from mapbuild.corpus import FORMS, TOPICS, build_prompts


def test_grid_is_complete_and_unique():
    items = build_prompts()
    assert len(items) == 120 == len(TOPICS) * len(FORMS)
    assert len({i["id"] for i in items}) == 120
    assert {i["form"] for i in items} == set(FORMS)
    assert all(i["topic"] in i["prompt"] for i in items)
