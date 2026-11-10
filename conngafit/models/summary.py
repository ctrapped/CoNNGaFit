# Helpers each CoNNGaFit 3-d network's summary() uses to print its structure:
#   - parameter counts per block and per layer
#   - the encoder/decoder rows shared by the U-Net networks' stage_shapes()
#   - the per-row table and the parameter-total footer printed by every summary()
#
# A summary row is a (label, n_features, dims) tuple; `groups` maps a row label to the list of
# modules whose parameters belong to that row.


def count_params(modules):
    """Number of parameters in a list of modules, counting each parameter once
    (a parameter can appear more than once if the modules overlap or share it)."""
    counted = set()
    total = 0
    for module in modules:
        for param in module.parameters():
            if id(param) not in counted:
                counted.add(id(param))
                total += param.numel()
    return total


def layer_param_lines(modules, indent="      "):
    """Text lines breaking the modules' parameters down by layer, for summary().

    One line per layer that has parameters, in aligned columns, e.g.
        conv1     Conv2d     params=296
        norm1     GroupNorm  params=16
    A module that is a single layer (e.g. a bare Conv2d) gives no lines.
    """
    layers = []
    for module in modules:
        for name, layer in module.named_modules():
            if name == "":  # named_modules() lists the module itself first, with an empty name
                continue
            n_params = sum(p.numel() for p in layer.parameters(recurse=False))  # this layer's own parameters
            if n_params > 0:
                layers.append((name, type(layer).__name__, n_params))
    if not layers:
        return []

    name_width = max(len(name) for name, _, _ in layers)
    type_width = max(len(layer_type) for _, layer_type, _ in layers)
    lines = []
    for name, layer_type, n_params in layers:
        lines.append(f"{indent}{name.ljust(name_width)}  {layer_type.ljust(type_width)}  params={n_params:,}")
    return lines


def unet_stage_shapes(stem_shape, enc_channels, up_modes):
    """Summary rows for a U-Net's encoder and decoder stages, computed analytically (no forward pass).

    stem_shape: (nSpec, nX, nY) after the stem. enc_channels: channel count of each encoder stage.
    up_modes: Upsample mode of each decoder stage ('conv_transpose' or 'trilinear').

    Encoder stage 0 keeps the stem's shape and every later stage downsamples each axis to ceil(N/2).
    Decoder stage j upsamples (N -> 2N-1 for 'conv_transpose', N -> 2N for 'trilinear'), then
    _match_spatial realigns it to the matching encoder skip map, which sets the stage's output shape.
    The last row's dims are therefore the stage-0 encoder shape.
    """
    rows = []
    enc_shapes = []
    shape = tuple(stem_shape)
    for i, ch in enumerate(enc_channels):
        if i > 0:
            shape = tuple(-(-s // 2) for s in shape)  # ceil(N/2) downsample
        enc_shapes.append(shape)
        rows.append((f"encoder stage {i}" + (" (no downsample)" if i == 0 else ""), ch, shape))

    n_stages = len(enc_channels)
    for j, mode in enumerate(up_modes):
        enc_idx = n_stages - 2 - j  # decoder stage j pairs with this encoder skip
        out_ch = enc_channels[enc_idx]
        up_shape = tuple((2 * s - 1 if mode == "conv_transpose" else 2 * s) for s in shape)
        rows.append((f"decoder {j}: upsample", out_ch, up_shape))
        skip = enc_shapes[enc_idx]
        rows.append((f"decoder {j}: + skip(enc {enc_idx})", 2 * out_ch, skip))
        rows.append((f"decoder {j}: deconv block", out_ch, skip))
        shape = skip
    return rows


def print_stage_rows(rows, groups, row_note, label_width, print_fn=print, layers=True):
    """Print one summary line per (label, n_features, dims) row: its feature count, parameter count
    ('-' for rows without parameters), dims and an optional note.

    groups: dict mapping row label -> list of modules holding that row's parameters.
    row_note: callable label -> note string or None.
    label_width: width of the label column.
    layers: if True, list each row's layers and their parameter counts under it.
    """
    for label, feats, shape in rows:
        mods = groups.get(label, [])
        params = f"{count_params(mods):,}" if mods else "-"
        dims = " x ".join(str(s) for s in shape)
        note = row_note(label)
        suffix = f" {note}" if note else ""
        print_fn(f"  {label:<{label_width}}   features={feats:<6} params={params:<11} dims={dims}{suffix}")
        if layers:
            for line in layer_param_lines(mods):
                print_fn(line)


def print_param_totals(model, bar_width, print_fn=print):
    """Print the summary footer: total and trainable parameter counts, then the module tree."""
    n_all = count_params([model])
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_fn("-" * bar_width)
    print_fn(f"parameters: {n_all:,} total ({n_train:,} trainable)")
    print_fn("=" * bar_width)
    print_fn(str(model))
    print_fn("=" * bar_width)
