"""LoRA adapters for RL fine-tuning of a Lafa model.

``add_lora`` turns the backbone's target ``nn.Linear`` layers into
``LoRALinear`` in place, so parameter names and module paths stay as they are
(the base ``weight``/``bias`` keep their keys; ``lora_A``/``lora_B`` are new),
and freezes everything else. ``merged_state_dict`` folds the adapters into
the base weights for export: the saved model is a plain Lafa checkpoint that
the evaluation harness loads like a released one.
"""

import json
import math
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Linear):
    """W x + b + (alpha / r) B A x, with W and b frozen. B starts at zero, so
    the adapted layer starts equal to the base layer."""

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        return out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling

    def merged_weight(self):
        """W + (alpha / r) B A, computed in float32, in W's dtype."""
        delta = self.lora_B.float() @ self.lora_A.float()
        return (self.weight.float() + self.lora_scaling * delta).to(self.weight.dtype)


def add_lora(model, rank, alpha, targets=TARGETS):
    """Adapt the backbone's ``targets`` linear layers with rank-``rank`` LoRA
    and freeze every other parameter. Returns the adapted modules by name.

    A is initialized like nn.Linear's weight (Kaiming uniform) from the global
    RNG, so two models adapted after the same seed get the same A."""
    for p in model.parameters():
        p.requires_grad_(False)
    adapted = {}
    for name, module in model.backbone.named_modules():
        if type(module) is not nn.Linear or name.rsplit(".", 1)[-1] not in targets:
            continue
        w = module.weight
        module.__class__ = LoRALinear
        module.lora_A = nn.Parameter(torch.empty(rank, module.in_features, device=w.device,
                                                 dtype=w.dtype))
        nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
        module.lora_B = nn.Parameter(torch.zeros(module.out_features, rank, device=w.device,
                                                 dtype=w.dtype))
        module.lora_scaling = alpha / rank
        adapted[f"backbone.{name}"] = module
    if not adapted:
        raise ValueError(f"no linear layers named {targets} in the backbone")
    return adapted


@torch.no_grad()
def merged_state_dict(model):
    """The model's state dict with every LoRALinear folded into its base
    weight and the adapter tensors dropped."""
    state = model.state_dict()
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.weight"] = module.merged_weight()
            del state[f"{name}.lora_A"], state[f"{name}.lora_B"]
    return state


ADAPTER_FILE = "adapter.safetensors"
ADAPTER_CONFIG = "adapter.json"


def save_adapters(model, path, *, base_model, rank, alpha):
    """Write only the adapters (as the model holds them, bf16) and the
    settings needed to rebuild the model: base model, rank, alpha, targets."""
    os.makedirs(path, exist_ok=True)
    state = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()
             if k.endswith(("lora_A", "lora_B"))}
    save_file(state, os.path.join(path, ADAPTER_FILE))
    with open(os.path.join(path, ADAPTER_CONFIG), "w") as f:
        json.dump(dict(base_model=base_model, rank=rank, alpha=alpha, targets=list(TARGETS)),
                  f, indent=1)


@torch.no_grad()
def load_adapters(model, path):
    """Add adapters to ``model`` (a base model) as saved by ``save_adapters``."""
    with open(os.path.join(path, ADAPTER_CONFIG)) as f:
        cfg = json.load(f)
    add_lora(model, cfg["rank"], cfg["alpha"], tuple(cfg["targets"]))
    state = load_file(os.path.join(path, ADAPTER_FILE))
    params = dict(model.named_parameters())
    if set(state) != {n for n in params if n.endswith(("lora_A", "lora_B"))}:
        raise RuntimeError(f"{path}: adapters do not match the model's LoRA layers")
    for k, v in state.items():
        params[k].copy_(v)
    return cfg
