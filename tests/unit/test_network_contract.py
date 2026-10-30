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

# Every (network, head) pair each network supports, and those with a conv head
NETWORK_HEADS = [(module, head) for module, heads in HEADS.items() for head in heads]
NETWORK_CONV_HEADS = [(module, head) for module, head in NETWORK_HEADS if head != "fc"]
every_head = pytest.mark.parametrize(
    ("network", "head"), NETWORK_HEADS, ids=[f"{m.__name__}-{h}" for m, h in NETWORK_HEADS]
)
every_conv_head = pytest.mark.parametrize(
    ("network", "head"), NETWORK_CONV_HEADS, ids=[f"{m.__name__}-{h}" for m, h in NETWORK_CONV_HEADS]
)


@every_head
def test_default_output_is_one_value_per_pixel(network, make_model, make_cube, head):
    model = make_model(network, head=head)
    cube = make_cube()
    _, n_x, n_y = model.input_shape
    assert model(cube).shape == (len(cube), n_x * n_y)


def test_output_size_override(network, make_model, make_cube):
    cube = make_cube()
    assert make_model(network, output_size=OUTPUT_SIZE)(cube).shape == (len(cube), OUTPUT_SIZE)


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


@every_conv_head
def test_conv_heads_require_one_output_per_pixel(network, make_model, head):
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


@every_head
@pytest.mark.parametrize("predict_variance", [False, True])
def test_training_loss_reaches_every_parameter(network, make_model, make_cube, head, predict_variance):
    """The training loss (as in train_model_unet.py: MSE, or Gaussian NLL with predict_variance) reaches every
    parameter, including the log-variance output's."""
    model = make_model(network, head=head, predict_variance=predict_variance)
    cube = make_cube()
    target = torch.randn(len(cube), model.output_size)
    if predict_variance:
        mean, logvar = model(cube)
        assert mean.shape == logvar.shape == target.shape  # one mean and log-variance per output
        loss = nn.GaussianNLLLoss()(mean, target, logvar.exp())
    else:
        loss = nn.MSELoss()(model(cube), target)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name  # parameter is connected to the loss
        assert torch.isfinite(param.grad).all(), name  # no NaN/inf gradients


@every_head
def test_output_activation_bounds_predictions(network, make_model, make_cube, head):
    out = make_model(network, head=head, output_activation=nn.Tanh)(make_cube())
    assert out.abs().max() <= 1


@every_head
def test_eval_mode_is_deterministic(network, make_model, make_cube, head):
    model = make_model(network, head=head, fc_dropout_rate=DROPOUT, block_dropout_rate=DROPOUT).eval()
    cube = make_cube()
    with torch.no_grad():
        torch.testing.assert_close(model(cube), model(cube))


@every_head
def test_predict_with_uncertainty_samples_dropout(network, make_model, make_cube, head):
    model = make_model(network, head=head, fc_dropout_rate=DROPOUT).eval()
    cube = make_cube()
    mean, std = model.predict_with_uncertainty(cube, n_samples=N_SAMPLES)
    assert mean.shape == std.shape == (len(cube), model.output_size)  # one mean and spread per output
    assert (std > 0).any()  # dropout was active, so the samples differ
    assert not any(module.training for module in model.modules())  # model left fully in eval mode


@every_head
def test_predict_with_uncertainty_combines_variances(network, make_model, make_cube, head):
    model = make_model(network, head=head, fc_dropout_rate=DROPOUT, predict_variance=True).eval()
    _, aleatoric, epistemic, total = model.predict_with_uncertainty(make_cube(), n_samples=N_SAMPLES)
    torch.testing.assert_close(total**2, aleatoric**2 + epistemic**2)


@every_head
@pytest.mark.parametrize("predict_variance", [False, True])
def test_param_breakdown_accounts_for_every_parameter(network, make_model, head, predict_variance):
    model = make_model(network, head=head, predict_variance=predict_variance)
    assert sum(n for _, n in model.param_breakdown()) == sum(p.numel() for p in model.parameters())


@every_head
def test_summary_reports_parameter_count(network, make_model, head):
    model = make_model(network, head=head)
    lines = []
    model.summary(print_fn=lines.append)
    n_params = sum(p.numel() for p in model.parameters())
    assert any(f"{n_params:,} total" in line for line in lines)
