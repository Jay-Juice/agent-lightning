"""The recovery probe compares model state, excluding FSDP-only shard padding."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
source = Path(__file__).resolve().parents[2] / "research/checkpoint_roundtrip.py"
spec = importlib.util.spec_from_file_location("checkpoint_roundtrip_probe", source)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_tail_padding_is_not_logical_model_state():
    parameter = torch.nn.Parameter(torch.arange(11, dtype=torch.float32))
    parameter._shard_numel_padded = 3
    parameter._sharded_size = parameter.shape
    parameter._local_shard = parameter.data
    initial = probe.fingerprint(probe.logical_parameter_view(parameter))
    with torch.no_grad():
        parameter[-3:].add_(100)
    assert probe.fingerprint(probe.logical_parameter_view(parameter)) == initial
    padding_before = parameter[-3:].detach().clone()
    with torch.no_grad():
        probe.logical_parameter_view(parameter).add_(.01)
    assert probe.fingerprint(probe.logical_parameter_view(parameter)) != initial
    torch.testing.assert_close(parameter[-3:], padding_before, rtol=0, atol=0)


def test_zero_padding_and_regular_parameters_keep_all_elements():
    parameter = torch.nn.Parameter(torch.ones(3, 4))
    assert probe.logical_parameter_view(parameter).numel() == 12
    flat = torch.nn.Parameter(torch.ones(12))
    flat._shard_numel_padded = 0
    flat._sharded_size = flat.shape
    flat._local_shard = flat.data
    assert probe.logical_parameter_view(flat).numel() == 12


def test_invalid_padding_or_storage_is_rejected():
    parameter = torch.nn.Parameter(torch.ones(12))
    parameter._shard_numel_padded = 13
    with pytest.raises(ValueError, match="padding metadata"):
        probe.logical_parameter_view(parameter)
    parameter._shard_numel_padded = 1
    parameter._sharded_size = parameter.shape
    parameter._local_shard = parameter.data.clone()
    with pytest.raises(ValueError, match="local sharded"):
        probe.logical_parameter_view(parameter)
