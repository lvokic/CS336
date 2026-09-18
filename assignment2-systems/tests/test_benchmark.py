import torch

import pytest

from cs336_systems.benchmark import MODEL_CONFIGS, benchmark, get_model_config, profile


def test_handout_model_configs():
    assert get_model_config("SMALL") == MODEL_CONFIGS["small"]
    assert MODEL_CONFIGS["medium"].d_model == 1024
    assert MODEL_CONFIGS["10b"].num_heads == 36
    with pytest.raises(ValueError, match="Unknown model size"):
        get_model_config("tiny")


def test_benchmark_returns_requested_number_of_samples():
    result = benchmark(lambda: torch.arange(32).square().sum(), warmup_steps=1, measure_steps=3, device="cpu")

    assert len(result.samples_ms) == 3
    assert result.mean_ms > 0
    assert result.std_ms >= 0


def test_profile_exports_chrome_trace(tmp_path):
    trace_path = tmp_path / "trace.json"
    result = profile(lambda: torch.arange(32).square().sum(), warmup_steps=0, profile_steps=1, device="cpu", trace_path=trace_path)

    assert trace_path.is_file()
    assert "aten::" in result.table
    assert result.trace_path == trace_path
