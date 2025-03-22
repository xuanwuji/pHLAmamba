from typing import List
import math
from functools import partial
import json
import os
from collections import namedtuple
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


@dataclass
class MambaConfig:
    d_model: int = 2560
    n_layer: int = 6
    vocab_size: int = 33
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    dropout: float = 0.2


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, n_segments):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=1)  # token embedding
        self.seg_embed = nn.Embedding(n_segments, d_model, padding_idx=2)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        embedding = self.tok_embed(x) + self.seg_embed(seg)
        return self.norm(embedding)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def shift_pad(x, pos: List[int]):
    device = x.device
    shift_x = torch.zeros_like(x, device=device)
    batch, max_length, d_model = x.size()
    for i in range(batch):
        shift_x[i] = torch.cat((x[i][max_length - pos[i]:], torch.zeros([max_length - pos[i], d_model], device=device)),
                               dim=0)
    return shift_x


class BiDirectionMixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            vocab_size: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.embedding = Embedding(vocab_size=vocab_size, d_model=d_model, n_segments=3).to(device)
        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.forward_layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.backward_layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.hidden_fc = nn.ModuleList(
            [nn.Linear(2 * d_model, d_model, **factory_kwargs) for i in range(n_layer)]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, seg, pos, inference_params=None):
        hidden_states = self.embedding(x=input_ids, seg=seg)
        residual = None
        for f_layer, b_layer, h_fc in zip(
                self.forward_layers, self.backward_layers, self.hidden_fc
        ):
            hidden_states_f, residual_f = f_layer(
                hidden_states, residual, inference_params=inference_params
            )
            flip_shift_residual = shift_pad(residual.flip([1]), pos=pos) if residual is not None else None
            hidden_states_b, residual_b = b_layer(
                shift_pad(hidden_states.flip([1]), pos=pos), flip_shift_residual, inference_params=inference_params
            )
            hidden_states_b = shift_pad(hidden_states_b.flip([1]), pos=pos)
            residual_b = shift_pad(residual_b.flip([1]), pos=pos)
            hidden_states = h_fc(torch.cat([hidden_states_f, hidden_states_b], dim=-1))
            residual = 0.5 * (residual_f + residual_b)

        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


class MambaLMHeadModel(nn.Module, GenerationMixin):
    def __init__(
            self,
            config: MambaConfig,
            initializer_cfg=None,
            device=None,
            dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        factory_kwargs = {"device": device, "dtype": dtype}
        dropout = config.dropout

        super().__init__()
        self.backbone = BiDirectionMixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(d_model, **factory_kwargs),
            nn.Linear(d_model, 512, **factory_kwargs),
            nn.SELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1, **factory_kwargs)
        )
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, seg, pos, inference_params=None, num_last_tokens=0):
        hidden_states = self.backbone(input_ids, seg, pos, inference_params=inference_params)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        cls_hidden = hidden_states[:, 0, :]
        cls_logits = self.classifier(cls_hidden)
        CausalLMOutput = namedtuple("CausalLMOutput", ["lm_logits", "cls_logits", "hidden_states"])
        return CausalLMOutput(lm_logits=lm_logits, cls_logits=cls_logits, hidden_states=hidden_states)
