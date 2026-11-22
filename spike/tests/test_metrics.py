from latent_capture.metrics import summarize, verdict

RUNS = [
    {"mode": "off", "tokens": 100, "seconds": 5.0, "peak_gb": 21.0, "swap_gb_before": 0.0, "swap_gb_after": 0.0},
    {"mode": "on", "tokens": 100, "seconds": 5.5, "peak_gb": 22.0, "swap_gb_before": 0.0, "swap_gb_after": 0.0},
]


def test_summarize():
    s = summarize(RUNS)
    assert s["tps_off"] == 20.0
    assert round(s["tps_on"], 2) == 18.18
    assert round(s["speed_ratio"], 2) == 0.91
    assert s["peak_gb"] == 22.0
    assert s["swap_growth_gb"] == 0.0


def test_verdict_go():
    assert verdict(summarize(RUNS)) == "go"


def test_verdict_no_go_on_slow_capture():
    slow = [RUNS[0], {**RUNS[1], "seconds": 10.0}]
    assert verdict(summarize(slow)) == "no-go"


def test_verdict_no_go_on_memory():
    assert verdict(summarize([RUNS[0], {**RUNS[1], "peak_gb": 27.0}])) == "no-go"


def test_verdict_no_go_on_swap_growth():
    assert verdict(summarize([RUNS[0], {**RUNS[1], "swap_gb_after": 1.5}])) == "no-go"
