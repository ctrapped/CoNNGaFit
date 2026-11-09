"""networks: each network's hyperparameter keys, the config files that set them, and the model build_model makes."""

import tomllib

import pytest
from conftest import CONFIG_DIR

from conngafit.models.unet3d_ddpm import GaussianDiffusion
from conngafit.models.unet3d_resblock import NeuralNetwork
from conngafit.networks import HYPERPARAM_KEYS, OPTIONAL_HYPERPARAM_KEYS, build_model

CONFIG_FILES = sorted(CONFIG_DIR.glob("hyperparams_*.toml"))
# Big enough for the default four encoder stages, which the config files don't override
BUILD_SHAPE = (32, 32, 32)  # (nSpec, nY, nX)
# Overrides that make each config file's network small enough to build in a moment
SMALL_HYPERPARAMS = dict(n_filt0=2, k0=3, k1=3, n_fc=8, n_timesteps=10)
NETWORK_CLASSES = {"resblock": NeuralNetwork, "ddpm": GaussianDiffusion}


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda path: path.name)
def test_config_files_have_their_networks_keys(path):
    with open(path, "rb") as f:
        hyperparams = tomllib.load(f)
    network = hyperparams["network"]
    assert HYPERPARAM_KEYS[network] <= hyperparams.keys()  # every key the network needs
    assert hyperparams.keys() - HYPERPARAM_KEYS[network] <= OPTIONAL_HYPERPARAM_KEYS[network]  # and no others


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda path: path.name)
def test_build_model_matches_network(path):
    with open(path, "rb") as f:
        hyperparams = {**tomllib.load(f), **SMALL_HYPERPARAMS}
    model = build_model(hyperparams["network"], BUILD_SHAPE, hyperparams)
    assert isinstance(model, NETWORK_CLASSES[hyperparams["network"]])
    if isinstance(model, GaussianDiffusion):
        assert model.flat_labels  # samples and labels are flattened maps, as conngafit_datasets.py gives them
        assert model.denoiser.input_shape == BUILD_SHAPE  # built for the cube


def test_ddpm_rejects_radial_curves():
    with open(CONFIG_DIR / "hyperparams_ddpm.toml", "rb") as f:
        hyperparams = {**tomllib.load(f), **SMALL_HYPERPARAMS}
    with pytest.raises(ValueError, match="predicts maps"):
        build_model("ddpm", BUILD_SHAPE, hyperparams, output_size=10)


def test_mass_weight_power_is_resblock_only():
    assert "mass_weight_power" in OPTIONAL_HYPERPARAM_KEYS["resblock"]  # the resblock network may weight its loss
    # but the ddpm network's denoising loss is unweighted, so its files can't set it
    assert "mass_weight_power" not in HYPERPARAM_KEYS["ddpm"] | OPTIONAL_HYPERPARAM_KEYS["ddpm"]
