"""Behaviour shared by every regression network module: shapes, heads, gradients and uncertainty estimates."""

import pytest
import torch
from torch import nn

import unet3d_dilated_resnet
import unet3d_resblock

HEADS = {
    unet3d_resblock: ("fc", "conv", "conv_halfres"),
    unet3d_dilated_resnet: ("fc", "conv"),
}
OUTPUT_SIZE = 5
DROPOUT = 0.5
N_SAMPLES = 8


def pytest_generate_tests(metafunc):
    """Run every test that takes `head` once per (network, head) pair the network supports."""
    if "head" in metafunc.fixturenames:
        pairs = [(module, head) for module, heads in HEADS.items() for head in heads]
        metafunc.parametrize(("network", "head"), pairs, ids=[f"{m.__name__}-{h}" for m, h in pairs])


def test_default_output_is_one_value_per_pixel(network, make_model, make_cube, head):
    model = make_model(network, head=head)
    cube = make_cube()
    _, n_x, n_y = model.input_shape
    assert model(cube).shape == (len(cube), n_x * n_y)


def test_output_size_override(network, make_model, make_cube):
    cube = make_cube()
    assert make_model(network, output_size=OUTPUT_SIZE)(cube).shape == (len(cube), OUTPUT_SIZE)


def test_predict_variance_returns_mean_and_logvar(network, make_model, make_cube, head):
    model = make_model(network, head=head, predict_variance=True)
    cube = make_cube()
    mean, logvar = model(cube)
    assert mean.shape == logvar.shape == (len(cube), model.output_size)


def test_rejects_cube_of_wrong_shape(network, make_model, make_cube):
    model = make_model(network)
    n_spec, n_x, n_y = model.input_shape
    with pytest.raises(ValueError, match="does not match"):
        model(make_cube((n_spec, n_x + 1, n_y)))


def test_rejects_input_shape_without_three_axes(network, make_model):
    with pytest.raises(ValueError, match="input_shape"):
        make_model(network, input_shape=(12, 12))


def test_rejects_unknown_head(network, make_model):
    with pytest.raises(ValueError, match="head"):
        make_model(network, head="mlp")


def test_conv_heads_require_one_output_per_pixel(network, make_model, head):
    if head == "fc":
        pytest.skip("the fc head can predict any output_size")
    with pytest.raises(ValueError, match="output_size"):
        make_model(network, head=head, output_size=OUTPUT_SIZE)


@pytest.mark.parametrize(
    "n_fc, n_fc_layers, expected",
    [
        (8, 1, [8]),
        (8, 2, [8, 8]),
        ([8, 6], 1, [8, 6]),
        (0, 1, []),
    ],
)
def test_fc_hidden_widths(network, make_model, make_cube, n_fc, n_fc_layers, expected):
    model = make_model(network, n_fc=n_fc, n_fc_layers=n_fc_layers)
    assert model.fc_widths == expected  # hidden-layer widths resolved from n_fc/n_fc_layers
    assert model(make_cube()).shape[1] == model.output_size  # forward pass still works with those widths


def test_backward_reaches_every_parameter(network, make_model, make_cube, head):
    model = make_model(network, head=head)
    model(make_cube()).square().mean().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name  # parameter is connected to the output
        assert torch.isfinite(param.grad).all(), name  # no NaN/inf gradients


def test_gaussian_nll_reaches_every_parameter(network, make_model, make_cube, head):
    """The predict_variance training loss (as in train_model_unet.py) reaches both the mean and log-variance outputs."""
    model = make_model(network, head=head, predict_variance=True)
    mean, logvar = model(make_cube())
    nn.GaussianNLLLoss()(mean, torch.randn_like(mean), logvar.exp()).backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name  # parameter is connected to the loss
        assert torch.isfinite(param.grad).all(), name  # no NaN/inf gradients


def test_output_activation_bounds_predictions(network, make_model, make_cube, head):
    out = make_model(network, head=head, output_activation=nn.Tanh)(make_cube())
    assert out.abs().max() <= 1


def test_eval_mode_is_deterministic(network, make_model, make_cube, head):
    model = make_model(network, head=head, fc_dropout_rate=DROPOUT, block_dropout_rate=DROPOUT).eval()
    cube = make_cube()
    with torch.no_grad():
        torch.testing.assert_close(model(cube), model(cube))


def test_predict_with_uncertainty_samples_dropout(network, make_model, make_cube, head):
    model = make_model(network, head=head, fc_dropout_rate=DROPOUT).eval()
    cube = make_cube()
    mean, std = model.predict_with_uncertainty(cube, n_samples=N_SAMPLES)
    assert mean.shape == std.shape == (len(cube), model.output_size)  # one mean and spread per output
    assert (std > 0).any()  # dropout was active, so the samples differ
    assert not any(module.training for module in model.modules())  # model left fully in eval mode


def test_predict_with_uncertainty_combines_variances(network, make_model, make_cube, head):
    model = make_model(network, head=head, fc_dropout_rate=DROPOUT, predict_variance=True).eval()
    _, aleatoric, epistemic, total = model.predict_with_uncertainty(make_cube(), n_samples=N_SAMPLES)
    torch.testing.assert_close(total**2, aleatoric**2 + epistemic**2)


@pytest.mark.parametrize("predict_variance", [False, True])
def test_param_breakdown_accounts_for_every_parameter(network, make_model, head, predict_variance):
    model = make_model(network, head=head, predict_variance=predict_variance)
    assert sum(n for _, n in model.param_breakdown()) == sum(p.numel() for p in model.parameters())


def test_summary_reports_parameter_count(network, make_model, head):
    model = make_model(network, head=head)
    lines = []
    model.summary(print_fn=lines.append)
    n_params = sum(p.numel() for p in model.parameters())
    assert any(f"{n_params:,} total" in line for line in lines)
