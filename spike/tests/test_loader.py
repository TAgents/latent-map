from mlx_lm import generate


def test_model_has_three_loop_passes(loaded):
    model, _ = loaded
    text = model.args.text
    assert text.loop_config["num_passes"] == 3
    assert text.hidden_size == 2304
    assert len(text.loop_core) == 12


def test_generates_text(loaded):
    model, tokenizer = loaded
    out = generate(model, tokenizer, prompt="The capital of France is", max_tokens=8)
    assert out.strip()
