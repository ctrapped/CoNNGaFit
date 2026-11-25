# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `blocks.SelfAttention`: self-attention over every position of a feature map, voxels of a cube or pixels of a map
  (GroupNorm, 1x1 q/k/v, scaled dot-product attention and a zero-initialised output projection added to the input).
  The `resblock`, `ddpm`, `resblock_2d` and `ddpm_2d` networks add it after the residual blocks of the encoder and
  decoder stages listed in the optional `attention_stages` hyperparameter (default `[]`); the `ddpm` and `ddpm_2d`
  configs set `[2]`.
- `ResidualBlock` and `DeconvBlock` take an `architecture` argument: `"post_activation"` (default, the existing
  layout) or `"identity_skip"`, which adds each block's correction to an unchanged skip and zero-initialises its
  last norm so the block starts as its skip. The four networks of `networks.py` (including their map heads' 2-d
  blocks) set it for every block with the optional `block_architecture` hyperparameter (default
  `"post_activation"`). Models saved without the key rebuild with `"post_activation"`.
- Two networks that compress each pixel's spectrum first and then work in 2-d at full map resolution, in
  `conngafit/models/spectral_unet2d.py`: `network = "resblock_2d"` (`SpectralResNet2d`, maps with an optional 1σ,
  like `resblock`) and `network = "ddpm_2d"` (`SpectralDenoisingUNet2d`, the denoiser of the diffusion model, like
  `ddpm`). Each is a `blocks.SpectralEncoder` (`n_spectral_layers` strided 3-d convs halving the spectral axis,
  then a per-pixel attention pool over it), a 2-d `UNet` and a `MapHead2d`; `k0` is the encoder's spectral kernel
  and `k1` every spatial kernel. Configs `configs/hyperparams_resblock_2d.toml` (with gradient clipping at 1.0) and
  `configs/hyperparams_ddpm_2d.toml`, and experiments `experiments/resnet_2d_velocity_map/` and
  `experiments/ddpm_2d_velocity_map/`. `networks.DIFFUSION_NETWORKS` lists the networks trained by denoising.
- `conngafit/models/unet.py`: `UNet`, the residual U-Net body (encoder, decoder with skips, self-attention) of
  the four networks, for cubes (`n_dims=3`) or maps (`n_dims=2`). Its `forward` returns the output and every stage's
  output.
- `ResidualBlock`, `DeconvBlock`, `Upsample` and `SelfAttention` take `n_dims` (3 for cubes, the default; 2 for
  maps), and `ResidualBlock` projects its skip whenever the channel count changes.
- `blocks.MapHead2d`, the map head on 2-d features (2-d residual blocks, dropout and a 3x3 output conv), and
  `blocks.MapHead3d`, which first pools the spectral axis, resizes to the map grid and concatenates any extra maps.
- Optional `n_stages` and `n_head_blocks` hyperparameters for all four networks (U-Net stages and map-head blocks).
- `check_stage_compatibility` takes `stem_reduction` (1 for networks without a stem) and 2-d map shapes.
- Several targets at once: `target` in the datasets file may be a list (e.g. `["mass", "rotational_momentum"]`),
  and the ResBlock U-Net (and the dilated ResNet, through its new `n_maps` argument) predicts a map or radial
  curve of each from one shared network. The DDPM denoises the maps of every target together, so each sample is
  drawn from their joint posterior. Labels are `(n_targets, nY, nX)` maps or `(n_targets, n_bins)` radial curves,
  each target is scaled separately
  (`label_stats` is `(2, n_targets)`), and the checkpoint records the targets in order.
- Parallel data loading: `train(num_workers=...)` / `train_model_unet.py --num-workers` read and transform
  samples in worker processes while the GPU trains, into pinned memory copied without blocking. Defaults to up to
  8 of the job's CPUs on a GPU, 0 (the main process) on a CPU. Any number of workers trains the same network,
  which matches `num_workers=0` in the first epoch and differs slightly after (see the test).
- With predict_variance (or the DDPM), the validation and test log lines also give the percentage of pixels or
  rings whose predicted 1σ includes the label, alongside the accuracy (within 0.1 label standard deviations).
- Snapshots during training: `train(every=N, compare_examples=4)` / `train_model_unet.py
  --compare-every N` save the hdf5 prediction and comparison image (map or radial curve) of a fixed handful of
  training and validation images every N epochs, to watch them improve. Off by default.
- `use_model.predict(..., return_samples=True)` yields the DDPM's joint posterior samples in physical units.
- pytest suite for the network modules: shared blocks, both regression networks
  (fc and conv heads), and the DDPM denoiser and diffusion process.
- GitHub Actions workflow running the tests on Python 3.11 and 3.12, plus a ruff lint job.
- ruff linting/formatting configuration and pre-commit hooks.
- `pyproject.toml` with dependencies and a `[dev]` extra.
- Pull request and issue templates, `LICENSE` (MIT), `CITATION.cff` and this changelog.
- `examples/demo.py`: trains a small network on committed VOF observations of six synthetic
  galaxies (`examples/demo_data/`, regenerated by `examples/make_demo_data.py`) and predicts the
  rotation curve of one it hasn't seen, in a few minutes on a CPU. Its figure is in the README.
- `examples/demo_inference.py`: runs the committed, pre-trained demo network (`examples/demo_model/`)
  on the demo's test galaxy with `use_model.run_inferences`, in a few seconds.
- `examples/demo.py --predict map`: trains the demo network on the rotational velocity map (with a mass-weighted
  loss) and bins the predicted map into the rotation curve; its figure adds the true and predicted velocity maps,
  their difference and the prediction's signal-to-noise. `examples/demo_inference.py` runs map networks too.
- `experiments/`: scripts on the demo data for trying out newer features, which don't yet predict as well as the
  demo:
  - `resnet_maps/`: the ResBlock U-Net predicts mass and rotational momentum maps, and the curve is each ring's
    total momentum over its total mass, with the maps' 1σ propagated into it.
  - `ddpm_maps/`: the DDPM samples both maps from their joint posterior, so the curve's 1σ, the spread of the
    samples' curves, includes the correlation between the mass and momentum errors.
  - `ddpm_velocity_map/`: the DDPM samples the rotational velocity map.

  The map figures add each map's true and predicted values, how they differ (log10(predicted / true), or
  predicted − true for velocity) and signal-to-noise, and every figure draws each property within the same color
  limits.
- `examples/demo_utils.py` and `examples/demo_plots.py`: the settings, loaders, computations and plotting functions
  shared by the demo and the experiments.
- `mass_weight_power` hyperparameter: weights each pixel's (or ring's) loss by mass to that power,
  so velocity networks aren't dominated by faint halo gas. 0 (the default) is the unweighted loss,
  and 0.5 a good start. With `predict_variance` it weights the Gaussian NLL.
- `seed` hyperparameter: seeds Python, NumPy and PyTorch before training and is saved in the checkpoint,
  so runs with the same seed, data and settings on the same machine give the same network.
- `tests/integration/test_pipeline.py`: every output head (and radial curves) trains for two epochs on one demo
  image, writes a valid checkpoint, and predicts outputs of the expected size when reloaded.
- `tests/integration/test_demo.py` (marked `slow`): trains the demo on all its data and checks the
  losses fall and the test galaxy's rotation curve is predicted to within 10 km/s, for the new and committed
  networks. CI runs it on pull requests; `pytest -m "not slow"` skips it locally.
- `predict_variance` hyperparameter: the ResBlock U-Net also predicts each value's 1σ uncertainty,
  trained with `nn.GaussianNLLLoss`. The uncertainty is saved with the final-epoch snapshots and
  comparison plots, returned by `use_model.predict(..., return_std=True)` and written by `run_inferences`.
- Training records the MSE of the predicted mean (`training_mses`, `validation_mses` and `final_test_mse`).
- Tests for `predict_variance`: Gaussian NLL gradients for every network and head, pipeline runs with
  every output head, and calibration of the demo's predicted uncertainties.
- The conditional diffusion model (DDPM, `unet3d_ddpm.py`) in the training and inference pipeline, chosen
  with `network = "ddpm"` and configured by `configs/hyperparams_ddpm.toml`. It trains on the denoising
  loss, and each prediction is the mean of `n_samples` posterior samples, saved and plotted with their
  standard deviation as its uncertainty. `networks.py` lists each network's hyperparameter keys and builds
  either network for `train_model_unet.py` and `use_model.py`.
- `predict_every` hyperparameter: the training and validation sets are predicted, and so their accuracy
  and MSE scored, every `predict_every` epochs and on the final epoch.
- `datasets.radial_profile(..., values=...)` bins a given map, such as a network's prediction, in place
  of the file's annotation, weighted by the file's mass.
- `datasets.ring_index(shape, fov_kpc, n_bins)`: each pixel's ring, as `radial_profile` bins them.
- `tests/unit/test_networks.py` (each network's keys, its config files and `build_model`) and
  `tests/integration/test_pipeline_ddpm.py` (the DDPM's training and inference pipeline).

### Changed
- The DDPM denoisers have `encode(cube)` and `denoise(features, y_t, t)` methods, and `forward` runs both.
  `GaussianDiffusion.sample` encodes each cube once and runs only `denoise` at each reverse step. As a result,
  `ddpm_2d` runs its spectral encoder once per cube instead of n_samples × T times, which makes sampling faster.
   The samples are unchanged, and so is `ddpm`.
- **Breaking:** the DDPM's timestep conditioning is wider. The sinusoidal encoding defaults to 128 dims (optional
  `time_encoding_dim` hyperparameter) instead of `n_filt0`, and the learned embedding passed to every block is as
  wide as the widest encoder stage instead of `4 * n_filt0`. The blocks' `t_dim` argument is renamed
  `time_embed_dim`. DDPM checkpoints saved before this change no longer load.
- **Breaking:** `CoNNGaFitImageDataset` samples are `(image, label, loss weights, index)`, and the dataset takes a
  `mass_weight_power` option; the weights and index were added by a wrapper in the training script. The index
  names each saved output after the DataLoader shuffles the samples.
- `train_model.train` is split into module-level functions, each usable on its own: `check_hyperparams`,
  `check_datasets`, `training_statistics`, `calc_label_scaling`, `make_dataloaders`, `compute_loss`, `train_loop`,
  `evaluate` (the validation and test loops, merged), `save_batch_outputs`, `save_checkpoint` and `save_history`.
  `plotting.py` gains `save_correlation_plots`, `plot_loss` and `plot_history`. The test set is loaded before
  training, so a mistake in its sources is found right away, and `TrainingResult` also records each validation
  sample's accuracy every epoch and each test sample's accuracy. Training's outputs are unchanged.
- **Breaking:** `fits_to_conngafit`'s `image_dir` and `distance` arguments are renamed `fits_path` and `distance_kpc`
  (`distance_kpc` defaults to `None`).
- **Breaking:** the networks' hand-built `summary()`, `stage_shapes()` and `param_breakdown()` methods are replaced by
  `conngafit.networks.model_summary(model, input_shape)`, a `torchinfo.summary` table of the layers, their output
  shapes and parameter counts (`torchinfo` is a new dependency). Training logs it at the start of each run, and
  `examples/model_summaries.ipynb` prints it for each network.
- **Breaking:** the ResBlock U-Net's and dilated ResNet's `predict_with_uncertainty` method is replaced by the
  function `conngafit.mc_dropout.predict_with_mc_dropout(model, x, n_samples)`, which serves every regression
  network; the name distinguishes its Monte Carlo dropout spread from the 1σ trained with `predict_variance` and
  from the DDPM's posterior sampling (`GaussianDiffusion.predict_with_uncertainty`, unchanged).
- **Breaking:** the U-Net body of `resblock` and `ddpm` is the shared `unet.UNet`, so its parameters are named
  `unet.encoder.*`, `unet.decoder.*` (the ResBlock U-Net's `dc_layers`), `unet.upsamples.*` and
  `unet.encoder_attention.*` / `unet.decoder_attention.*`. Networks of the same seed and settings are unchanged, but
  checkpoints saved before no longer load. The committed demo network is retrained.
- **Breaking:** the map heads' 2-d blocks are `ResidualBlock(n_dims=2)` instead of `ResBlock2d`: without time
  conditioning (the `conv` head) they use BatchNorm2d instead of GroupNorm, every block applies its activation after
  the skip addition, and a projected skip has a norm. `ConvMapHead` and the DDPM's `MapHead` are merged into
  `blocks.MapHead3d` with the same parameter names, so `conv`-head and DDPM checkpoints saved before the block
  change no longer load.
- **Breaking:** `Upsample`'s `"trilinear"` mode is renamed `"interpolation"`, since it is bilinear for maps.
- `DeconvBlock` picks BatchNorm without time conditioning and GroupNorm with it, as `ResidualBlock` does, instead of
  relying on its caller to pass `max_groups=None`.
- **Breaking:** the ResBlock U-Net's and dilated ResNet's `fc_dropout_rate`, `fc_activation` and `pool_output`
  arguments are renamed `head_dropout_rate`, `head_activation` and `fc_pool_shape` (the ResBlock U-Net's only), since
  the first two also apply to the conv heads. `head_activation` now also sets the `conv` head's activation; its
  default keeps every head's activation unchanged (ReLU for `conv`, LeakyReLU otherwise). Hyperparameter files are
  unchanged.
- **Breaking:** the datasets file's optional `base_dir` key is renamed `data_dir`.
- Training is the function `conngafit.train_model.train(hyperparams, datasets, ...)`, taking the parameters as dicts
  and returning the network's path and loss history; `scripts/train_model_unet.py` loads the two files and calls it.
  Checkpoints save the parameters as TOML written from the dicts (`hyperparams_text`, `datasets_text`), without the
  files' comments.
- `examples/demo.py` walks through training from Python: it sets every hyperparameter and the datasets as dicts and
  calls `train`, instead of writing parameter files and running the training script.
- The demo saves only its network, loss history and figure; `--diagnostics` also saves the training
  script's snapshots and diagnostic plots, which `train_model_unet.py --no-diagnostics` turns off.
- **Breaking:** the library modules are the installable `conngafit` package, with the networks in
  `conngafit/models/`, and the command-line scripts are in `scripts/`; install with `pip install -e .`
  before running them. Imports use the package paths (e.g. `from conngafit.use_model import predict`).
  Renamed: `conngafit_datasets` → `datasets`, `plotting_functions` → `plotting`, `unet_blocks` → `blocks`,
  `unet_diagnostics` → `diagnostics`, `inference_template.py` → `run_inference.py`. The Quartz guide moved from
  `Tutorials/Quartz/` to `docs/quartz/`.
- **Breaking:** maps stay 2-d through the whole pipeline instead of being flattened: datasets return
  `(n_targets, nY, nX)` labels, the ResBlock U-Net and dilated ResNet return `(nBatch, n_maps, H, W)` maps (or
  `(nBatch, n_maps, output_size)` radial curves; `output_size` is now for 1-d outputs of the fc head only), the
  DDPM's denoiser takes and returns `(nBatch, n_maps, H, W)` maps and samples `(n_samples, nBatch, n_maps, H, W)`,
  and `use_model.predict` yields `(n_targets, ...)` arrays, also for a single target. `GaussianDiffusion`'s
  `flat_labels`, `plotting.make_image`'s `target_shape` and `plotting.save_hdf5_1d` are removed. Checkpoints and
  saved outputs are unchanged.
- Saved outputs include the target in their names: `<sample>_<target>_Epoch<N>` for training snapshots, and
  `<name>_<target>_projectionMap` / `_radialPlot` for `run_inferences`.
- **Breaking:** the mass-flux targets are now momentum targets, mass × velocity in Msun km/s:
  `radial_momentum` and `cylindrical_radial_momentum` (formerly `radial_mass_flux` and
  `cylindrical_radial_mass_flux`), plus the new `rotational_momentum`. Divide by a pixel or ring
  width after inference to get a mass flux.
- **Breaking:** hyperparameter files need a `seed` key (`seed = 0` in `configs/`).
- **Breaking:** hyperparameter files need a `predict_variance` key (`predict_variance = false` in `configs/`).
- **Breaking:** hyperparameter files need `network` and `predict_every` keys, and are checked against their
  network's keys (`predict_variance` and `mass_weight_power` are ResBlock only). Checkpoints save every
  hyperparameter of their network, and `use_model.py` needs their `network`.
- **Breaking:** `configs/hyperparams.toml` and `configs/hyperparams_rc.toml` are renamed
  `configs/hyperparams_resblock.toml` and `configs/hyperparams_resblock_rc.toml`.
- **Breaking:** `unet3d_ddpm.DenoisingUNet3d` takes `input_shape` (and optional `map_shape`) and checks
  them at construction, like `unet3d_resblock.NeuralNetwork`. `GaussianDiffusion` samples by DDPM ancestral sampling
  through every timestep.
- The demo trains with `predict_variance`. Its figure adds ±1σ bands, a residual panel and both loss
  curves, and the committed demo network is retrained.
- Loss plots use a log axis unless the loss can be negative.
- The demo tests check the MSE of the mean, since the Gaussian NLL can be negative.
- Tests are split into `tests/unit/` (networks and building blocks, a few seconds) and
  `tests/integration/` (training, inference and the demo); `pytest tests/unit` runs only the unit tests.
- **Breaking:** `use_model.py` runs networks trained by `train_model_unet.py`, rebuilding them from
  their checkpoint; `run_inferences` and the inference scripts no longer take a network type.
- Validation accuracy counts predictions within 0.1 standard deviations of the training labels,
  instead of one standard deviation of each batch.
- Radial-curve comparison plots are plotted against radius in kpc, with a labeled colorbar.

### Removed
- `blocks.ResBlock2d` and `blocks.ConvMapHead` (replaced by `ResidualBlock(n_dims=2)` and `MapHead3d`), and
  `blocks.stem_out_shape`, whose only caller was `check_stage_compatibility`.
- `inference_test_validation_set.py`, which ran a network on the validation set listed in an annotations CSV; the
  datasets no longer use annotation CSVs. Run `use_model.run_inferences` on the validation images instead.

### Fixed
- `fits_to_conngafit` wrote an all-zero cube when the FITS cube had as many channels as the network takes, or when
  its image didn't bin evenly into a grid larger than the network's, crashed when the cube had an odd number of
  channels more than the network takes, and saved a negative `pixel_res_kpc` without a distance. It now converts the
  cube from Jy/beam to HI column density per channel (cm^-2, optically thin, from the header's beam and channel
  width), the units of the training cubes; averages the image in blocks with `astropy.nddata.block_reduce` (it
  summed them, scaling the column density by the number of pixels per block); and crops or zero-pads each axis,
  centered, to the network's grid. `pixel_res_kpc` is saved only with a distance.
- The per-snapshot validation accuracy plot was empty unless training ran as a single batch.
- Diagnostic plot names: "Valdiation" → "Validation".
