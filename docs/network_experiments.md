# Networks and tunable knobs

A reference for planning training experiments: every network in the repo (including ones not yet wired into
training), and every setting that can be changed on each. Settings are grouped by how you reach them:

- **Config**: a key in the hyperparameters TOML (or the `HYPERPARAMS` dict passed to `train()`), checked by
  `conngafit/networks.py`. Change it and rerun.
- **Constructor only**: an argument of the network's class with a default, but no config key. To experiment with
  it, add it to `NETWORK_KEYS` / `OPTIONAL_HYPERPARAM_KEYS` in `networks.py` (on `main`, optional keys there are
  passed straight to the constructor).
- **Code**: fixed in the code; needs an edit.

Keys marked **(main)** exist on `main` (merged from `unet_self_att`) but not yet on `training-diagnostic-enhancements`.

## The networks at a glance

| Network | File | `network =` | Predicts | Uncertainty | Status |
|---|---|---|---|---|---|
| Residual U-Net | `models/unet3d_resblock.py` | `"resblock"` | maps or radial curves | `predict_variance` (Gaussian NLL), MC dropout | In use |
| Conditional DDPM | `models/unet3d_ddpm.py` | `"ddpm"` | maps only | posterior spread of `n_samples` draws | Hooked up; experiments in `experiments/ddpm_maps`, `experiments/ddpm_velocity_map` |
| Spectral 2-d ResNet | `models/spectral_unet2d.py` | `"resblock_2d"` | maps only | `predict_variance` (Gaussian NLL), MC dropout | Hooked up; experiment in `experiments/resnet_2d_velocity_map` |
| Spectral 2-d DDPM | `models/spectral_unet2d.py` | `"ddpm_2d"` | maps only | posterior spread of `n_samples` draws | Hooked up; experiment in `experiments/ddpm_2d_velocity_map` |
| Dilated ResNet | `models/unet3d_dilated_resnet.py` | none | maps or radial curves | `predict_variance`, MC dropout | **Not hooked up**: not in `networks.py` |

All of them take a cube of shape `(nSpec, nX, nY)` and predict `n_maps` outputs, one per target in `datasets["target"]`.

---

## 1. Residual U-Net (`resblock`)

A stem (stride-2 conv + stride-2 max pool, every axis to ceil(N/4)), an encoder of `n_stages` stages of two
residual blocks (channels `n_filt0 * 2**i`, each later stage halving every axis), a decoder of `n_stages - 1`
stages with skip connections, and one of three output heads.

For a 128³ cube: stem → 32³; encoder stages 0–3 at 32³, 16³, 8³, 4³ with `n_filt0`, 2×, 4×, 8× channels.

### Config

| Key | Values / default | What it changes | Notes for experiments |
|---|---|---|---|
| `n_filt0` | int (configs: 6) | Channels of encoder stage 0; doubles each stage | Main capacity knob. Memory and compute grow ~quadratically in it |
| `k0` | int (configs: 9) | Kernel of the stem conv (same on all 3 axes) | Larger sees more of the line profile before the 4× reduction |
| `k1` | int (configs: 3) | Kernel of every residual/deconv block | 3 is standard; 5 grows the receptive field at ~4.6× the cost per block |
| `output_head` | `"fc"`, `"conv"`, `"conv_halfres"` | Output head (below) | Radial curves need `"fc"` |
| `n_fc` | int or list (default 64; rc config 3000) | Width of the fc head's hidden layer(s) | fc head only. Output Linear has `n_fc × n_maps × prod(output_shape)` weights |
| `dropout_rate` | 0–1 (configs: 0.3) | Dropout in the head | First regularizer to reach for |
| `block_dropout_rate` | 0–1 (default 0) | Channel-wise `Dropout3d` in every residual/deconv block | Broad regularizer; start at 0.1–0.15 |
| `predict_variance` | bool | Also predict a per-value log-variance; train with Gaussian NLL | Needs `label_scaling` ≠ `"none"`. Changes the accuracy metric to "within 1σ" |
| `mass_weight_power` | ≥ 0 (default 0) | Weight each value's loss by mass^p (normalized per image) | 0.5 for curves, 1 for velocity maps in our runs; pixels with no mass are not trained |
| `block_architecture` **(main)** | `"post_activation"` (default), `"identity_skip"` | Residual block layout | `identity_skip` starts each block as the identity (zero-init second norm), puts dropout between the convs, and has no activation after the add; often trains more stably in deeper nets |
| `attention_stages` **(main)** | list of stage indices, default `[]` | `SelfAttention` after the listed encoder stages and the decoder stage at the same resolution | Cost ∝ (voxels at that stage)². For 128³ cubes stage 3 (64 voxels) and 2 (512) are cheap, 1 (4096) is heavy, 0 (32768) is impractical |
| `n_stages` | int ≥ 2 (default 4) | U-Net encoder depth, checked against the cube size | Fewer stages: smaller receptive field and bottleneck; more needs a larger cube |
| `n_head_blocks` | int (default 2) | 2-d residual blocks in the `conv` head | |

### Output heads

| Head | How it works | Use for |
|---|---|---|
| `fc` | Adaptive average pool to `fc_pool_shape` (8×8×8), flatten, `n_fc_layers` hidden Linear layers, output Linear | Radial curves, small maps (≈40×40). Grows with the output size |
| `conv` | `MapHead3d`: `SpectralAttentionPool` (learned per-pixel weights over velocity channels), bilinear resize to the map grid, `n_head_blocks` 2-d residual blocks, 3×3 conv | Large maps; size independent of map size |
| `conv_halfres` | One more decoder stage up to the stem conv's half resolution (with a skip), spectral axis folded into channels, the bottleneck's global average appended to every pixel, two 3×3 convs (`head_channels`), 1×1 conv, resize | Large maps where pixel detail matters; the most expensive head |

### Constructor only

| Argument | Default | What it changes |
|---|---|---|
| `kernel_stem`, `kernel_coder` | `(k0,)*3`, `(k1,)*3` | 3-tuples, so **anisotropic kernels** are possible, e.g. longer along the spectral axis: `(9, 5, 5)`. `build_model` currently makes them isotropic |
| `n_fc_layers` | 1 | Hidden layers in the fc head (ignored if `n_fc` is a list) |
| `fc_pool_shape` | (8, 8, 8) | Grid the fc head pools to; sets the fc input size `n_filt0 × prod(shape)` |
| `head_channels` | 64 | Width of the `conv_halfres` head's convs |
| `upsample_mode` | `"conv_transpose"` | Or `"interpolation"` (trilinear interpolation + 1×1×1 conv): avoids checkerboard artifacts in maps |
| `stem_activation` | `nn.LeakyReLU` | |
| `block_activation` | `nn.ReLU` | e.g. `nn.SiLU`, `nn.GELU` |
| `head_activation` | ReLU (`conv`), LeakyReLU (others) | |
| `output_activation` | `nn.Identity` | e.g. `nn.Tanh` to bound outputs to the scaled label range [−1, 1] |
| `SelfAttention(n_heads)` **(main)** | 1 (hard-coded in the U-Net) | Attention heads; must divide the stage's channels |

### Not hooked up

- `conngafit.mc_dropout.predict_with_mc_dropout(model, x, n_samples=30)`: Monte Carlo dropout, for every regression
  network. Runs the network `n_samples` times with dropout on and returns the mean and the spread across passes (the
  epistemic uncertainty). With `predict_variance` it also returns the aleatoric σ and the combined σ. Neither
  `use_model.py` nor training calls it. Needs `dropout_rate` or `block_dropout_rate` > 0 to give a spread.
- `forward(x, save_latent_images=True)`: saves the spectrally summed activations at every stage as PNGs.

---

## 2. Conditional DDPM (`ddpm`)

`GaussianDiffusion` wraps `DenoisingUNet3d`, a U-Net like the ResBlock one whose input is the cube plus the noisy
maps (tiled along the spectral axis). Every block uses GroupNorm and is conditioned on the timestep with FiLM.
`MapHead3d` re-injects the noisy maps at full resolution. Training regresses the added noise (MSE). Prediction draws
`n_samples` maps by running all `n_timesteps` reverse steps, then returns their mean and standard deviation. Maps
only: no radial curves, no `mass_weight_power`, no `predict_variance`.

### Config

| Key | Values / default | What it changes | Notes for experiments |
|---|---|---|---|
| `n_filt0`, `k0`, `k1`, `block_dropout_rate` | as for resblock | Denoiser capacity and regularization | |
| `n_timesteps` | int (config: 1000) | Diffusion steps T | Prediction cost ∝ T. A checkpoint only loads with the same T and schedule |
| `schedule` | `"cosine"` (config), `"linear"` | Noise schedule | Cosine destroys signal more gradually; better for small, low-resolution maps |
| `n_samples` | int (config: 30) | Posterior draws per prediction | Cost ∝ n_samples. More draws give a less noisy mean and σ |
| `label_scaling` | `"symmetric"` / `"one-sided"` | Must map labels to [−1, 1] to match the noise; predicted clean maps are clipped to [−1, 1] | `"none"` turns clipping off |
| `predict_every` | int (config: 500) | How often training predicts (scores accuracy and MSE) | Each prediction costs `n_samples × n_timesteps` denoiser passes per cube |
| `time_encoding_dim` **(main)** | 128 | Width of the sinusoidal timestep encoding | On `main` the learned embedding is as wide as the widest encoder stage |
| `block_architecture`, `attention_stages` **(main)** | as for resblock | | `hyperparams_ddpm.toml` on `main` sets `attention_stages = [2]` |
| `n_stages` | int ≥ 2 (default 4) | Denoiser depth | |
| `n_head_blocks` | int ≥ 0 (default 2) | Full-resolution 2-d blocks in the head (`MapHead3d`) | |

### Constructor only

| Argument | Default | What it changes |
|---|---|---|
| `upsample_mode` | `"interpolation"` | Avoids checkerboard artifacts, which look like structured noise to the denoiser |
| `max_groups` | 8 | Upper bound on GroupNorm groups |
| `stem_activation`, `block_activation` | LeakyReLU, SiLU | |
| `GaussianDiffusion(clip_range)` | (−1, 1) | Bounds on the predicted clean map during sampling |
| `sample(max_batch=...)` | all cubes | Caps cubes per denoiser call during sampling, to bound memory |

### Not hooked up / elsewhere

- **Faster sampling**: there is only full DDPM sampling. A comment in `unet3d_ddpm.py` notes DDIM (fewer,
  deterministic steps, with `eta`) as a to-do. It would cut prediction cost by 10–50×.
- **Diagnostics** (`origin/add_ddpm_diagnostics`, unmerged): `train(diagnostic_mode=True)` records the loss by
  timestep, the reverse chain, the calibration of the posterior σ, and how concentrated the spectral attention is.

---

## 3. Spectral 2-d ResNet (`resblock_2d`)

`SpectralResNet2d` works on maps after its first stage. A `SpectralEncoder` compresses each pixel's spectrum:
`n_spectral_layers` strided 3-d convs with kernel `(k0, k1, k1)`, each halving the spectral axis and doubling the
channels, then a `SpectralAttentionPool` over what is left of it. A 3×3 conv brings the features to `n_filt0`
channels, a 2-d `UNet` of `n_stages` stages refines them at the **full map resolution**, and `MapHead2d` predicts the
maps. The U-Net and head use BatchNorm (no time conditioning); the encoder uses GroupNorm. Maps only, with the same
`predict_variance` and MC dropout as the ResBlock U-Net.

For a 128³ cube with the config: encoder 128 → 64 → 32 → 16 spectral channels at 128×128, pooled to
`4 × n_filt0` features per pixel; U-Net stages 0–4 at 128², 64², 32², 16², 8² with `n_filt0`, 2×, 4×, 8×, 16×
channels.

### Config

| Key | Values / default | What it changes | Notes for experiments |
|---|---|---|---|
| `n_filt0` | int (config: 6) | Channels of the encoder's first layer, the U-Net's first stage and the head | Main capacity knob |
| `k0` | int (config: 9) | Spectral kernel of the encoder's convs | Sets how much of each line profile one conv sees |
| `k1` | int (config: 3) | Spatial kernel of every conv (the encoder's x and y, the U-Net and the head) | |
| `n_spectral_layers` | int (default 3) | Encoder layers; the spectral axis is pooled after ceil(nSpec / 2^n) channels | More layers: a longer spectral receptive field and wider features |
| `n_stages` | int ≥ 2 (default 5) | 2-d U-Net depth, checked against nX and nY | Works at full resolution, so it needs more stages than the 3-d U-Net for the same receptive field |
| `n_head_blocks` | int ≥ 0 (default 2) | 2-d residual blocks in `MapHead2d` | |
| `dropout_rate` | 0–1 (config: 0.3) | `Dropout2d` before the head's output conv | |
| `block_dropout_rate` | 0–1 (default 0) | `Dropout2d` in every U-Net block | |
| `predict_variance`, `mass_weight_power` | as for resblock | | |
| `block_architecture` | as for resblock | Every U-Net and head block | |
| `attention_stages` | list, default `[]` | `SelfAttention` over every pixel after the listed stages | Cost ∝ (pixels at that stage)². For 128² maps stage 2 (1024 pixels) is cheap, 1 (4096) is heavy, 0 (16384) is impractical |
| `max_grad_norm` | config: 1.0 | Gradient clipping | On in the config: an unclipped prototype diverged on one seed of three |

`output_head` and `n_fc` don't apply: there is one map head and no radial curves.

### Constructor only

| Argument | Default | What it changes |
|---|---|---|
| `upsample_mode` | `"conv_transpose"` | Or `"interpolation"` (bilinear interpolation + 1×1 conv) |
| `activation` | `nn.ReLU` | Every layer's activation |
| `max_groups` | 8 | Upper bound on GroupNorm groups in the encoder and in self-attention |
| `output_activation` | `nn.Identity` | As for resblock |

`forward(x, save_latent_images=True)` is not supported.

---

## 4. Spectral 2-d DDPM (`ddpm_2d`)

`GaussianDiffusion` wraps `SpectralDenoisingUNet2d`: the same spectral encoder, after which the noisy maps are
concatenated to the encoder features at full resolution, a 3×3 conv to `n_filt0`, a 2-d `UNet` and a `MapHead2d`
whose output conv starts at zero. Every U-Net and head block uses GroupNorm and FiLM on the timestep embedding (as
wide as the widest U-Net stage). Training and prediction are those of the DDPM. The head doesn't re-inject the noisy
maps, since they already enter the U-Net at full resolution.

### Config

| Key | Values / default | What it changes | Notes for experiments |
|---|---|---|---|
| `n_filt0`, `k0`, `k1`, `n_spectral_layers`, `n_stages`, `n_head_blocks`, `block_dropout_rate` | as for resblock_2d | Denoiser capacity and regularization | |
| `n_timesteps`, `schedule`, `n_samples`, `label_scaling`, `predict_every`, `time_encoding_dim` | as for ddpm | | |
| `block_architecture` | as for resblock | | |
| `attention_stages` | list (config: `[2]`) | As for resblock_2d | |

### Constructor only

| Argument | Default | What it changes |
|---|---|---|
| `upsample_mode` | `"interpolation"` | As for ddpm |
| `activation` | `nn.SiLU` | Every layer's activation |
| `max_groups` | 8 | Upper bound on GroupNorm groups |

---

## 5. Dilated ResNet (not hooked up)

"Encode-Process-Decode" (Stachenfeld et al. 2021): one conv into feature space, `n_blocks` dilated residual blocks
at constant channels and **full resolution** (no down/upsampling, no U-Net skips; the receptive field grows by
dilation), one conv out, then the same `fc` or `conv` heads as the ResBlock U-Net (no `conv_halfres`). It has the
same `predict_variance` and MC dropout support.

To experiment with it: add a `"dilated"` entry to `HYPERPARAM_KEYS` and `build_model` in `networks.py`. Check
`use_model.load_model` too.

| Argument | Default | What it changes | Notes |
|---|---|---|---|
| `n_filt0` | required | Channels, **constant** through the whole network | Full-resolution features are expensive: 128³ × `n_filt0` floats per layer |
| `kernel_stem` | required | Encoder and decoder conv kernels | |
| `kernel_coder` | required | Every dilated conv's kernel | |
| `n_blocks` | 4 | Dilated residual blocks | 4 blocks × 7 rates = 28 conv layers |
| `dilation_rates` | (1, 2, 4, 8, 4, 2, 1) | Dilations cycled in every block | Sets the receptive field; with k=3 one block sees ±22 voxels (±44 after two blocks) |
| `stem_stride` | (1, 1, 1) | Downsampling in the encoder conv | **Use (2,2,2) or more for 128³ cubes**, or memory will not fit |
| `n_fc`, `n_fc_layers`, `fc_pool_shape` | 64, 1, (8,8,8) | fc head | |
| `head_dropout_rate`, `block_dropout_rate` | 0, 0 | | |
| `encode_activation`, `decode_activation` | None | | |
| `block_activation`, `head_activation`, `output_activation` | ReLU, ReLU/LeakyReLU, Identity | | |
| `predict_variance`, `head`, `output_shape`, `n_head_blocks`, `n_maps` | as for resblock | | |

---

## 6. Training knobs (every network)

### Config

| Key | Values / default | Notes |
|---|---|---|
| `learning_rate` | e.g. 1e-4 | Constant: **no warm-up or scheduler** exists |
| `weight_decay` | e.g. 1e-3 | With `adam`, an L2 term on every parameter; with `adamw`, decoupled, conv/linear weights only (try 1e-2) |
| `optimizer` | `"adam"` (default), `"adamw"` | |
| `ema_decay` | default 0.999; 0 = raw weights | Evaluation, prediction and the saved model all use the EMA. It averages ~1/(1 − decay) steps; keep that well below the total number of steps. The demos train for ~2200 steps (200 epochs × 11 batches): with 0.999 the 2-d ResNet demo's curve RMS was 105 km/s, with 0.99 13 km/s, so the demos set 0.99 |
| `max_grad_norm` | 0 = off | Set to ~2–3× the typical norm in the log, so only spikes are clipped. Check `clipped in N of M batches` |
| `epochs` | | The final epoch's EMA weights are saved (no best-epoch checkpoint) |
| `predict_every` | ≥ 1 | Only affects how often accuracy/MSE are scored (and cost) |
| `label_scaling` | `"none"`, `"symmetric"` (÷ max\|label\|), `"one-sided"` (min/max → [−1, 1]) | Symmetric for signed targets (radial momentum); one-sided for rotational velocity |
| `seed` | int | Weight init, dropout and batch order. Vary it to measure run-to-run scatter before trusting small differences |

### `train()` arguments / script flags

| Argument | Default | Notes |
|---|---|---|
| `batch_size` | ~1/10 of the training set (≥ 2); 0 = full batch | BatchNorm in the ResBlock U-Net needs ≥ 2. Changes the effective step size and noise |
| `test_batch_size` | ~1/10 of the validation set | Speed only |
| `num_workers` | up to 8 CPUs on a GPU, 0 on CPU | Speed only; same results |
| `save_diagnostics` | True (`train_model_unet.py`), off unless `--diagnostics` (examples) | Final-epoch snapshots, comparison images, loss/accuracy/correlation plots |
| `snapshot_every`, `snapshot_examples` | 0 (off), 4 | Comparison images of a fixed set of training and validation images every N epochs |

### Fixed in code

- `ACCURACY_TOLERANCE = 0.1` label standard deviations, used for the accuracy without an uncertainty.
- Adam betas and eps are the PyTorch defaults.
- No mixed precision, no early stopping, no LR schedule or warm-up.

---

## 7. Dataset knobs (`datasets` dict / `datasets.toml`)

| Key | Values | Notes |
|---|---|---|
| `target` | one of `mass`, `rotational_velocity`, `radial_velocity`, `cylindrical_radial_velocity`, `rotational_momentum`, `radial_momentum`, `cylindrical_radial_momentum`, or a list | A list predicts every target from one shared network (multi-task) |
| `radial_bins` | 0 = maps, N = radial curve of N rings | Curves need `output_head = "fc"` |
| `radial_projection` | `"sky"`, `"face_on"` | Where the rings are measured |
| `inclinations` | `"all"` or a list, e.g. `[45]` | The same for training, validation and testing. A network trained on one inclination does poorly on others |
| `augment` | `"none"`, `"major_axis"` (×4), `"all"` (×8) | Rotations and flips of the training images only |
| `training`, `validation`, `testing` | `{galaxy dir: snapshot or [min, max]}` | Keep the test galaxy (and its physics variants, e.g. `cosmic_ray/` and `mhd/` versions of the same halo) out of training |
| `data_dir`, `suffix` | optional | |

---

## 8. Retired variants (git history)

The 2023–24 networks in `NetworkModels/AlternativeNetworks/` were removed in `153b117`; recover one with
`git show 153b117^:NetworkModels/AlternativeNetworks/<file>.py`. They were built for 40×40×77 cubes and
are superseded by the networks above, but they record ideas that were tried:

| Idea | Files |
|---|---|
| Tanh output activation (bounded outputs) | `Unet3d_TanhEnd`, `Unet3d_TanhNew`, `Unet3d_TanhSimple`, `Unet3d_Tanh07242024` (now `output_activation`) |
| ReLU vs LeakyReLU throughout | `Unet3d_Relu07242024`, `Unet3d_Leaky` (now the `*_activation` arguments) |
| No high-resolution skip connections | `Unet3d_NoHighFeatureforwarding`, `Unet3d_NoHighFeatureForwardingNew` (`feature_forwarding` is the hook for this) |
| No residual connections | `Unet3d_NoRes` |
| Two fully connected layers | `Unet3d_2FC` (now `n_fc_layers` / `n_fc` as a list) |
| Moment maps as extra inputs | `Unet3d_WithMomentMaps`, `Unet3d_TanhEnd_WithMomentMaps` (no current equivalent) |
| Time-averaged output | `Unet3d_TimeAveragedOutput` |
