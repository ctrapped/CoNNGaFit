"""Capturing the outputs of a network's internal layers, for diagnostics of what they compute."""

from contextlib import contextmanager


@contextmanager
def captured_outputs(modules):
    """Collect the output of each named module every time it runs, inside the with block.

    modules: dict of name -> nn.Module. Yields a dict of name -> list of that module's outputs (detached tensors),
    one per call in order. The forward hooks are removed on leaving the block, even after an error.
    """
    outputs = {name: [] for name in modules}

    def hook_for(name):
        def hook(module, inputs, output):
            outputs[name].append(output.detach())

        return hook

    handles = [module.register_forward_hook(hook_for(name)) for name, module in modules.items()]
    try:
        yield outputs
    finally:
        for handle in handles:
            handle.remove()
