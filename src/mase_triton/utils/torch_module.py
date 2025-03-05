import torch


def get_layer_name(module: torch.nn.Module, layer: str) -> str:
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is layer:
            return name
    raise ValueError(f"Cannot find op {layer} in module {module}")


def get_layer_by_name(module: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == layer_name:
            return m
    raise ValueError(f"Cannot find op {layer_name} in module {module}")


def set_layer_by_name(module: torch.nn.Module, name: str, new_layer: torch.nn.Module) -> None:
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = module
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_layer)
    else:
        setattr(module, name, new_layer)
