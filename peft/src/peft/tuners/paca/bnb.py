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

import warnings
from typing import Any, Optional

import bitsandbytes as bnb
import torch

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_bnb_weight, gather_params_ctx
from peft.utils.other import transpose

from .layer import PacaLayer
from torch.cuda.amp import custom_fwd, custom_bwd

import torch.nn as nn

import bitsandbytes.functional as BF

if is_bnb_4bit_available():

    class Linear4bit(torch.nn.Module, PacaLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            r: int = 0,
            paca_alpha: int = 1,
            gradient_accumulation_steps: int = 1,
            **kwargs,
        ) -> None:
            super().__init__()
            PacaLayer.__init__(self, base_layer)
            self.fan_in_fan_out = False

            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                r,
                paca_alpha=paca_alpha,
                )
        
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
            self.scaling[adapter_name] = paca_alpha / r
            
            # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
            with gather_params_ctx(self.get_base_layer().weight):
                self.paca_init(adapter_name)
        
        
        def paca_init(self, adapter_name):
            weight = self.get_base_layer().weight
            output = dequantize_bnb_weight(weight, state=weight.quant_state)
            #self.paca_w[adapter_name].data = self.prune_weight(self.weight, self.paca_indices).to(self.paca_w[adapter_name].dtype)*1/self.scaling[adapter_name]
            self.paca_w[adapter_name].data = prune_weight(output, self.paca_indices[adapter_name])*1/self.scaling[adapter_name]
        
        
        def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
            """
            Merge the active adapter weights into the base weights

            Args:
                safe_merge (`bool`, *optional*):
                    If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                    before merging the weights. This is useful if you want to check if the merge operation will produce
                    NaNs. Defaults to `False`.
                adapter_names (`list[str]`, *optional*):
                    The list of adapter names that should be merged. If None, all active adapters will be merged.
                    Defaults to `None`.
            """
            adapter_names = check_adapters_to_merge(self, adapter_names)
            if not adapter_names:
                # no adapter to merge
                return

            for active_adapter in adapter_names:
                if active_adapter not in self.paca_w.keys():
                    continue

                warnings.warn(
                    "Merge lora module to 4-bit linear may get different generations due to rounding errors."
                )
                # Refer to https://gist.github.com/ChrisHayduk/1a53463331f52dca205e55982baf9930
                weight = self.get_base_layer().weight
                kwargs = weight.__dict__
                
                output = dequantize_bnb_weight(weight, state=weight.quant_state)
                output.data[:,self.paca_indices[active_adapter]] = self.scaling[active_adapter]*self.paca_w[active_adapter].to(output.dtype)
                
                
                if safe_merge and not torch.isfinite(output).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                if "bnb_quantized" in kwargs:
                    kwargs["bnb_quantized"] = False
                kwargs["requires_grad"] = False
                kwargs.pop("data", None)
                self.get_base_layer().weight = bnb.nn.Params4bit(output.to("cpu"), **kwargs).to(weight.device)
                self.merged_adapters.append(active_adapter)

        
        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            self._check_forward_args(x, *args, **kwargs)
            adapter_names = kwargs.pop("adapter_names", None)

            if adapter_names is not None:
                result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                # result = self.base_layer(x, *args, **kwargs)
                # As per Tim Dettmers, for 4bit, we need to defensively clone here.
                # The reason is that in some cases, an error can occur that backprop
                # does not work on a manipulated view. This issue may be solved with
                # newer PyTorch versions but this would need extensive testing to be
                # sure.
                # result = result.clone()
                weight = self.get_base_layer().weight
                
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.paca_w.keys():
                        continue
                    paca_w = self.paca_w[active_adapter]*self.scaling[active_adapter]
                    result = pacalinear.apply(x, weight, paca_w, self.bias, self.paca_indices[active_adapter], weight.quant_state)
            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "paca." + rep
        
        
    class pacalinear(torch.autograd.Function):
        # Note that both forward and backward are @staticmethods
        @staticmethod
        @custom_fwd
        # bias is an optional argument
        def forward(ctx, input, weight, paca_w=None, bias=None, paca_indices=None, quant_state=None):
            dweight=dequantize_bnb_weight(weight, quant_state).to(input.dtype)
            dweight.data[:,paca_indices] = paca_w.to(input.dtype)
            
            if input.dim() == 2 and bias is not None:
                # fused op is marginally faster
                ret = torch.addmm(bias, input, dweight.t())
            else:
                output = input.matmul(dweight.t())
                if bias is not None:
                    output += bias
                ret = output
            ctx.save_for_backward(prune_activation(input, paca_indices), weight, bias, paca_w, paca_indices)
            ctx.quant_state=quant_state
            return ret

        # This function has only a single output, so it gets only one gradient
        @staticmethod
        @custom_bwd
        def backward(ctx, grad_output):
            pruned_input, weight, bias, paca_w, paca_indices = ctx.saved_tensors
            grad_input = grad_paca_w = grad_bias = None
            
            dweight=dequantize_bnb_weight(weight, ctx.quant_state).to(grad_output.dtype)
            dweight.data[:,paca_indices] = paca_w.to(grad_output.dtype)
            
            if ctx.needs_input_grad[0]:
                grad_input = grad_output.matmul(dweight)
                    
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
            return grad_input, None, grad_paca_w, grad_bias, None, None


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


def dispatch_bnb_4bit(target: torch.nn.Module, adapter_name: str, **kwargs):
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    loaded_in_4bit = kwargs.get("loaded_in_4bit", False)
    if loaded_in_4bit and is_bnb_4bit_available() and isinstance(target_base_layer, bnb.nn.Linear4bit):
        fourbit_kwargs = kwargs.copy()
        fourbit_kwargs.update(
            {
                "compute_dtype": target_base_layer.compute_dtype,
                "compress_statistics": target_base_layer.weight.compress_statistics,
                "quant_type": target_base_layer.weight.quant_type,
            }
        )
        new_module = Linear4bit(target, adapter_name, **fourbit_kwargs)

    return new_module