# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose

from .config import PacaConfig

from .._buffer_dict import BufferDict
from torch.cuda.amp import custom_fwd, custom_bwd

class PacaLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("paca_w",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "paca_alpha", "scaling")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.paca_alpha = {}
        self.scaling = {}
        self.paca_w = nn.ParameterDict({})
        self.paca_indices = BufferDict({}, persistent=True)
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self._caches: dict[str, Any] = {}
        self.kwargs = kwargs
        
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, adapter_name, r, paca_alpha,
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.paca_alpha[adapter_name] = paca_alpha
        
        self.paca_w[adapter_name] = nn.Parameter(torch.zeros(self.out_features, r))      
        # Register buffer
        self.paca_indices[adapter_name] = torch.randperm(self.in_features)[:r]
        #self.paca_indices[adapter_name] = torch.topk(self.weight.norm(dim=0), r).indices
        #self.paca_indices[adapter_name] = torch.argsort(self.weight.norm(dim=0))[:r]
        self.scaling[adapter_name] = paca_alpha / r
        
        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        with gather_params_ctx(self.get_base_layer().weight):
            self.paca_init(adapter_name)
        

    def paca_init(self, adapter_name):
        weight = self.get_base_layer().weight
        dtype = weight.dtype
        
        #self.paca_w[adapter_name].data = self.prune_weight(self.weight, self.paca_indices).to(self.paca_w[adapter_name].dtype)*1/self.scaling[adapter_name]
        self.paca_w[adapter_name].data = prune_weight(self.weight, self.paca_indices[adapter_name])*1/self.scaling[adapter_name]
        
    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.paca_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.paca_w.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.paca_w.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.paca_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

        unique_adapters = set(self.active_adapters)
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = "Cannot pass `adapter_names` when DoRA is enabled."
                raise ValueError(msg)
    

# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, PacaLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        paca_alpha: int = 1,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        gradient_accumulation_steps: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        PacaLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            paca_alpha=paca_alpha,
            )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.iterations=0

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.paca_w.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights.data[:,self.paca_indices[active_adapter]] = self.scaling[active_adapter]*self.paca_w[active_adapter].to(orig_weights.dtype)
                
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data[:,self.paca_indices[active_adapter]] = self.scaling[active_adapter]*self.paca_w[active_adapter].to(base_layer.weight.dtype)
                    
                self.merged_adapters.append(active_adapter)
                
    def paca_update(self, adapter_name):
        with torch.no_grad():
            self.weight.data[:,self.paca_indices[adapter_name]] = self.scaling[adapter_name]*self.paca_w[adapter_name].to(self.weight.dtype)
                
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
    
        if adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            self.iterations += 1
            for active_adapter in self.active_adapters:
                if active_adapter not in self.paca_w.keys():
                    continue
                if self.training and self.iterations % (2*self.gradient_accumulation_steps):
                    self.paca_update(active_adapter)
                paca_w = self.paca_w[active_adapter]*self.scaling[active_adapter]
                result = pacalinear.apply(x, self.weight, paca_w, self.bias, self.paca_indices[active_adapter])
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "paca." + rep


class pacalinear(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    @custom_fwd
    # bias is an optional argument
    def forward(ctx, input, weight, paca_w=None, bias=None, paca_indices=None):
        if input.dim() == 2 and bias is not None:
            # fused op is marginally faster
            ret = torch.addmm(bias, input, weight.t())
        else:
            output = input.matmul(weight.t())
            if bias is not None:
                output += bias
            ret = output
        ctx.save_for_backward(prune_activation(input, paca_indices), weight, bias)
        return ret

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        pruned_input, weight, bias = ctx.saved_tensors
        grad_input = grad_paca_w = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
                
        if ctx.needs_input_grad[2]:
            dim = grad_output.dim()
            if dim > 2:
                grad_paca_w = grad_output.reshape(-1,
                                                  grad_output.shape[-1]).t().matmul(pruned_input.reshape(-1, pruned_input.shape[-1]))
            else:
                grad_paca_w = grad_output.t().matmul(pruned_input)

        if bias is not None and ctx.needs_input_grad[2]:
            if dim > 2:
                grad_bias = grad_output.sum([i for i in range(dim - 1)])
            else:
                grad_bias = grad_output.sum(0)
        return grad_input, None, grad_paca_w, grad_bias, None


def prune_weight(weight, indices=None):
    if indices is None:
        raise ValueError("Indices must be provided to prune activation dimensions.")
    pruned_weight = weight[:, indices]
    return pruned_weight

def prune_activation(activation, indices=None):
    if indices is None:
        raise ValueError("Indices must be provided to prune activation dimensions.")
    pruned_activation = activation[:, :, indices]
    return pruned_activation

def dispatch_default(target: torch.nn.Module, adapter_name: str, **kwargs):
    new_module = None
    
    bias = kwargs.pop("bias", False)
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    new_module = Linear(
            target,
            adapter_name,
            bias=bias,
            **kwargs,
        )
    return new_module