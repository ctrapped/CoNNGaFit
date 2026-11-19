"""Monte Carlo dropout uncertainty for the regression networks."""

import torch
from torch import nn


def predict_with_mc_dropout(model, x, n_samples=30, **forward_kwargs):
    """Estimate how uncertain a trained regression network is about x with Monte Carlo dropout: run it n_samples
    times with only its dropout layers in training mode, so each pass drops different channels, and take the mean
    and spread of the results. Every other layer stays in eval mode, so BatchNorm keeps using the statistics stored
    during training rather than those of the (often single-cube) batch it is given.

    This is separate from predict_variance. predict_variance trains the network to predict each value's 1-sigma, the
    noise in the data (aleatoric uncertainty). The spread here measures how much the prediction depends on which
    channels dropout removes, the network's own uncertainty (epistemic). It needs a network trained with dropout
    (head_dropout_rate or block_dropout_rate > 0); otherwise every pass is identical and the spread is 0.

    model: a regression network whose forward returns the mean, or (mean, logvar) when model.predict_variance.
    forward_kwargs: passed to every forward call (e.g. save_latent_images).

    Returns (mean, epistemic_std), each (nBatch, n_maps, *output_shape). With predict_variance, returns
    (mean, aleatoric_std, epistemic_std, total_std): aleatoric_std is the network's predicted 1-sigma averaged over
    the passes, and total_std = sqrt(aleatoric_std**2 + epistemic_std**2). Leaves the network in eval mode.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()
    with torch.no_grad():
        outputs = [model(x, **forward_kwargs) for _ in range(n_samples)]
    model.eval()

    if model.predict_variance:
        means = torch.stack([mean for mean, _ in outputs], dim=0)
        logvars = torch.stack([logvar for _, logvar in outputs], dim=0)
        mean_pred = means.mean(dim=0)
        aleatoric_std = logvars.exp().mean(dim=0).sqrt()
        epistemic_std = means.std(dim=0)
        total_std = (aleatoric_std**2 + epistemic_std**2).sqrt()
        return mean_pred, aleatoric_std, epistemic_std, total_std
    preds = torch.stack(outputs, dim=0)
    return preds.mean(dim=0), preds.std(dim=0)
