"""The networks train_model_unet.py trains and use_model.py runs, named by a hyperparameter file's network key:
"resblock" - the residual U-Net in unet3d_resblock.py, predicting maps or radial curves
"ddpm"     - the conditional diffusion model in unet3d_ddpm.py, predicting maps
"""

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

CLIP_RANGE = (-1.0, 1.0)  # Where scaled labels lie; the ddpm network's samples are clipped to it


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
            fc_dropout_rate=hyperparams["dropout_rate"],
            block_dropout_rate=hyperparams["block_dropout_rate"],
            head=hyperparams["output_head"],
            output_size=output_size,
            predict_variance=hyperparams["predict_variance"],
            n_maps=n_maps,
        )

    if output_size is not None:
        raise ValueError(
            "the ddpm network predicts maps of size equal to the input pixel shape (nY, nX). output_shape must be set to None."
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
        flat_labels=True,  # Labels are flattened maps, as conngafit_datasets.py gives them
        clip_range=None if hyperparams["label_scaling"] == "none" else CLIP_RANGE,
    )
