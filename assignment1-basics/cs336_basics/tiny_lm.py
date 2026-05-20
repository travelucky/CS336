from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from cs336_basics.model import transformer_lm


@dataclass(frozen=True)
class TinyTransformerConfig:
    vocab_size: int
    context_length: int = 64
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    d_ff: int = 192
    rope_theta: float = 10000.0
    init_std: float = 0.02


def _parameter_key(weight_name: str) -> str:
    return weight_name.replace(".", "__dot__")


def _weight_names(num_layers: int) -> list[str]:
    names = ["token_embeddings.weight"]
    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}"
        names.extend(
            [
                f"{prefix}.attn.q_proj.weight",
                f"{prefix}.attn.k_proj.weight",
                f"{prefix}.attn.v_proj.weight",
                f"{prefix}.attn.output_proj.weight",
                f"{prefix}.ln1.weight",
                f"{prefix}.ffn.w1.weight",
                f"{prefix}.ffn.w2.weight",
                f"{prefix}.ffn.w3.weight",
                f"{prefix}.ln2.weight",
            ]
        )
    names.extend(["ln_final.weight", "lm_head.weight"])
    return names


class TinyTransformerLM(nn.Module):
    """Thin trainable wrapper around the function-style Assignment 1 Transformer."""

    def __init__(self, config: TinyTransformerConfig):
        super().__init__()
        self.config = config
        self._names = _weight_names(config.num_layers)
        self.params = nn.ParameterDict()
        self._init_parameters()

    def _init_parameters(self) -> None: # 权值的模块大小
        cfg = self.config
        matrices: dict[str, tuple[int, ...]] = {
            "token_embeddings.weight": (cfg.vocab_size, cfg.d_model),
            "lm_head.weight": (cfg.vocab_size, cfg.d_model),
        }
        for layer_idx in range(cfg.num_layers):
            prefix = f"layers.{layer_idx}"
            matrices.update(
                {
                    f"{prefix}.attn.q_proj.weight": (cfg.d_model, cfg.d_model),
                    f"{prefix}.attn.k_proj.weight": (cfg.d_model, cfg.d_model),
                    f"{prefix}.attn.v_proj.weight": (cfg.d_model, cfg.d_model),
                    f"{prefix}.attn.output_proj.weight": (cfg.d_model, cfg.d_model),
                    f"{prefix}.ffn.w1.weight": (cfg.d_ff, cfg.d_model),
                    f"{prefix}.ffn.w2.weight": (cfg.d_model, cfg.d_ff),
                    f"{prefix}.ffn.w3.weight": (cfg.d_ff, cfg.d_model),
                }
            )

        norm_weights = {"ln_final.weight": (cfg.d_model,)}
        for layer_idx in range(cfg.num_layers):
            norm_weights[f"layers.{layer_idx}.ln1.weight"] = (cfg.d_model,)
            norm_weights[f"layers.{layer_idx}.ln2.weight"] = (cfg.d_model,)

        for name in self._names:
            if name in matrices:
                value = torch.empty(matrices[name])
                nn.init.normal_(value, mean=0.0, std=cfg.init_std)
            else:
                value = torch.ones(norm_weights[name])
            self.params[_parameter_key(name)] = nn.Parameter(value)

    @property
    def config_dict(self) -> dict[str, int | float]:
        return asdict(self.config)

    def assignment_weights(self) -> dict[str, torch.Tensor]:
        return {name: self.params[_parameter_key(name)] for name in self._names}

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        return transformer_lm(
            vocab_size=cfg.vocab_size,
            context_length=cfg.context_length,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            d_ff=cfg.d_ff,
            rope_theta=cfg.rope_theta,
            weights=self.assignment_weights(),
            in_indices=token_ids,
        )
