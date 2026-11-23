from mapbuild.naming import answer_text, name_area, score_name


def test_name_area_cleans_reply_and_refuses_thin_areas():
    assert name_area(lambda p: '"Python function syntax."\nBecause...', ["def", "return", "self"]) == "Python function syntax"
    assert name_area(lambda p: "x", ["def", "return"]) == ""


def test_score_name_counts_correct_letters():
    own, others = ["def", "return"], [["cat", "dog"], ["one", "two"], ["red", "blue"]]

    def oracle(prompt):
        for line in prompt.splitlines():
            if "def" in line and line[:1] in "ABCD":
                return line[0]
        return "?"

    assert score_name(oracle, "code", own, others) == 1.0
    assert score_name(lambda p: "I think it is none of them", "code", own, others) == 0.0
    assert score_name(oracle, "", own, others) == 0.0


def test_answer_text_keeps_only_the_answer_section():
    raw = "<|message_model|><|content_thinking|>The user wants a name.<|end_message|><|message_model|><|content_text|>Python code<|end_message|>"
    assert answer_text(raw) == "Python code"
    assert answer_text("<|message_model|><|content_thinking|>still thinking when cut off") == ""
    assert answer_text("plain reply") == "plain reply"


def test_letter_parsing_reads_ordinary_reply_styles():
    from mapbuild.naming import pick_letter

    assert [pick_letter(r) for r in ("B", "b)", "(B)", "**B**", "Answer: B", "The answer is B.")] == ["B"] * 6
    assert pick_letter("All of them") is None and pick_letter("I think a dog") is None and pick_letter("") is None
