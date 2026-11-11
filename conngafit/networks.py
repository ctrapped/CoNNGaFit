"""The networks train_model.train trains and use_model.py runs, named by the hyperparameters' network key:
"resblock" - the residual U-Net in unet3d_resblock.py, predicting maps or radial curves
"ddpm"     - the conditional diffusion model in unet3d_ddpm.py, predicting maps
"""

import torch
import torchinfo

from conngafit.models.unet3d_ddpm import DenoisingUNet3d, GaussianDiffusion
from conngafit.models.unet3d_resblock import NeuralNetwork

# Keys every hyperparameter file must have, and may have, for each network (see configs/hyperparams_<network>*.toml)
SHARED_KEYS = {
    "network",
    "learning_rate",
    "weight_decay",
    "epochs",
    "n_filt0",
    "k0",
    "k1",
    "block_dropout_rate",
    "label_scaling",
    "predict_every",
    "seed",
}
HYPERPARAM_KEYS = {
    "resblock": SHARED_KEYS | {"n_fc", "dropout_rate", "output_head", "predict_variance"},
    "ddpm": SHARED_KEYS | {"n_timesteps", "schedule", "n_samples"},
}
OPTIONAL_HYPERPARAM_KEYS = {"resblock": {"mass_weight_power"}, "ddpm": set()}


def build_model(network, input_shape, hyperparams, output_size=None, n_maps=1):
    """The network for cubes of input_shape (nSpec, nY, nX), predicting a map, or a radial curve of output_size bins
    (resblock only), for each of n_maps targets."""
    kernel_stem = (hyperparams["k0"],) * 3
    kernel_coder = (hyperparams["k1"],) * 3
    if network == "resblock":
        return NeuralNetwork(
            input_shape,
            hyperparams["n_filt0"],
            kernel_stem,
            kernel_coder,
            hyperparams["n_fc"],
            head_dropout_rate=hyperparams["dropout_rate"],
            block_dropout_rate=hyperparams["block_dropout_rate"],
            head=hyperparams["output_head"],
            output_size=output_size,
            predict_variance=hyperparams["predict_variance"],
            n_maps=n_maps,
        )

    if output_size is not None:
        raise ValueError(
            "the ddpm network predicts maps of size equal to the input pixel shape (nY, nX). output_size must be set to None."
        )
    denoiser = DenoisingUNet3d(
        input_shape,
        hyperparams["n_filt0"],
        kernel_stem,
        kernel_coder,
        block_dropout_rate=hyperparams["block_dropout_rate"],
        n_maps=n_maps,
    )
    return GaussianDiffusion(
        denoiser,
        n_timesteps=hyperparams["n_timesteps"],
        schedule=hyperparams["schedule"],
        flat_labels=True,  # Labels are flattened maps, as datasets.py gives them
        clip_range=None if hyperparams["label_scaling"] == "none" else (-1.0, 1.0),
    )


def model_summary(model, input_shape, depth=3):
    """A table of a network's layers with each one's output shape and parameter count (torchinfo.summary), from one
    forward pass on a blank cube of input_shape (nSpec, nY, nX). For the DDPM, the table is of its denoiser.
    depth sets how many levels of nested layers are listed."""
    cube = torch.zeros(1, *input_shape)
    if isinstance(model, GaussianDiffusion):
        noisy_maps = torch.zeros(1, model.n_maps, *model.map_shape)
        timestep = torch.zeros(1, dtype=torch.long)
        return torchinfo.summary(
            model.denoiser, input_data=[cube, noisy_maps, timestep], depth=depth, mode="eval", verbose=0
        )
    return torchinfo.summary(model, input_data=[cube], depth=depth, mode="eval", verbose=0)
