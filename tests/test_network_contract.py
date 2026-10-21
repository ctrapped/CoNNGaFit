"""Behaviour shared by every network module: shapes, heads, gradients and uncertainty estimates."""
import pytest
import torch
from torch import nn

OUTPUT_SIZE = 5
DROPOUT = 0.5
N_SAMPLES = 8


def test_default_output_is_one_value_per_pixel(network, make_model, make_cube):
    model = make_model(network)
    cube = make_cube()
    _, nX, nY = model.input_shape
    assert model(cube).shape == (len(cube), nX * nY)


def test_output_size_override(network, make_model, make_cube):
    cube = make_cube()
    assert make_model(network, output_size=OUTPUT_SIZE)(cube).shape == (len(cube), OUTPUT_SIZE)


def test_predict_variance_returns_mean_and_logvar(network, make_model, make_cube):
    model = make_model(network, predict_variance=True)
    cube = make_cube()
    mean, logvar = model(cube)
    assert mean.shape == logvar.shape == (len(cube), model.output_size)


def test_rejects_cube_of_wrong_shape(network, make_model, make_cube):
    model = make_model(network)
    nSpec, nX, nY = model.input_shape
    with pytest.raises(ValueError, match="does not match"):
        model(make_cube((nSpec, nX + 1, nY)))


def test_rejects_input_shape_without_three_axes(network, make_model):
    with pytest.raises(ValueError, match="input_shape"):
        make_model(network, input_shape=(12, 12))


@pytest.mark.parametrize("nFC, n_fc_layers, expected", [
    (8, 1, [8]),
    (8, 2, [8, 8]),
    ([8, 6], 1, [8, 6]),
    (0, 1, []),
])
def test_fc_hidden_widths(network, make_model, make_cube, nFC, n_fc_layers, expected):
    model = make_model(network, nFC=nFC, n_fc_layers=n_fc_layers)
    assert model.fc_widths == expected
    assert model(make_cube()).shape[1] == model.output_size


def test_backward_reaches_every_parameter(network, make_model, make_cube):
    model = make_model(network)
    model(make_cube()).square().mean().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_output_activation_bounds_predictions(network, make_model, make_cube):
    out = make_model(network, output_activation=nn.Tanh)(make_cube())
    assert out.abs().max() <= 1


def test_eval_mode_is_deterministic(network, make_model, make_cube):
    model = make_model(network, fc_dropout_rate=DROPOUT, block_dropout_rate=DROPOUT).eval()
    cube = make_cube()
    with torch.no_grad():
        torch.testing.assert_close(model(cube), model(cube))


def test_predict_with_uncertainty_samples_dropout(network, make_model, make_cube):
    model = make_model(network, fc_dropout_rate=DROPOUT).eval()
    cube = make_cube()
    mean, std = model.predict_with_uncertainty(cube, n_samples=N_SAMPLES)
    assert mean.shape == std.shape == (len(cube), model.output_size)
    assert (std > 0).any()
    assert not any(module.training for module in model.modules())


def test_predict_with_uncertainty_combines_variances(network, make_model, make_cube):
    model = make_model(network, fc_dropout_rate=DROPOUT, predict_variance=True).eval()
    _, aleatoric, epistemic, total = model.predict_with_uncertainty(make_cube(), n_samples=N_SAMPLES)
    torch.testing.assert_close(total**2, aleatoric**2 + epistemic**2)


def test_summary_reports_parameter_count(network, make_model):
    model = make_model(network)
    lines = []
    model.summary(print_fn=lines.append)
    n_params = sum(p.numel() for p in model.parameters())
    assert any(f"{n_params:,} total" in line for line in lines)
