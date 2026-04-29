#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""
VPP (Virtual Pipeline Parallelism) model patches for text-only Llama models.

This patch intentionally targets only the dense text-only LlamaModel and
LlamaForCausalLM path used by Llama 3.1 style models. Multimodal Llama
wrappers such as Mllama need separate model-specific VPP handling.
"""
from __future__ import annotations

from itertools import islice

from torch import nn
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_ascend.distributed.parallel_state import (
    get_virtual_pipeline_parallel_rank,
    get_virtual_pipeline_parallel_size,
)
from vllm_ascend.distributed.vpp_utils import (
    is_vpp_first_stage,
    is_vpp_last_stage,
    make_vpp_layers,
)


def _get_vp_size() -> int:
    from vllm_ascend.ascend_config import get_ascend_config

    try:
        return get_ascend_config().virtual_pipeline_parallel_size
    except RuntimeError:
        return 1


def _get_custom_layer_ranges_for_rank() -> list[tuple[int, int]] | None:
    from vllm_ascend.ascend_config import get_ascend_config

    try:
        all_ranges = get_ascend_config().vpp_layer_ranges
    except RuntimeError:
        return None
    if all_ranges is None:
        return None
    return all_ranges[get_pp_group().rank_in_group]


def _vpp_is_first_stage() -> bool:
    vp_size = get_virtual_pipeline_parallel_size()
    if vp_size <= 1:
        return get_pp_group().is_first_rank
    pp_rank = get_pp_group().rank_in_group
    vp_stage = get_virtual_pipeline_parallel_rank()
    return is_vpp_first_stage(pp_rank, vp_stage)


def _vpp_is_last_stage() -> bool:
    vp_size = get_virtual_pipeline_parallel_size()
    if vp_size <= 1:
        return get_pp_group().is_last_rank
    pp_group = get_pp_group()
    return is_vpp_last_stage(
        pp_group.rank_in_group,
        pp_group.world_size,
        get_virtual_pipeline_parallel_rank(),
        vp_size,
    )


def _vpp_rank_has_output(pp_rank: int, pp_size: int, vp_size: int) -> bool:
    if vp_size <= 1:
        return get_pp_group().is_last_rank
    if vp_size % 2 == 0:
        return pp_rank == 0
    return pp_rank == pp_size - 1


def _vpp_rank_has_embedding(
    pp_rank: int,
    pp_size: int,
    vp_size: int,
    *,
    tie_word_embeddings: bool,
) -> bool:
    if vp_size <= 1:
        return get_pp_group().is_first_rank
    return pp_rank == 0 or (
        tie_word_embeddings and _vpp_rank_has_output(pp_rank, pp_size, vp_size)
    )


def _vpp_stage_bounds(self) -> tuple[bool, bool, int, int]:
    if not hasattr(self, "vpp_layer_ranges"):
        return (
            get_pp_group().is_first_rank,
            get_pp_group().is_last_rank,
            self.start_layer,
            self.end_layer,
        )

    vp_stage = get_virtual_pipeline_parallel_rank()
    start, end = self.vpp_layer_ranges[vp_stage]
    return _vpp_is_first_stage(), _vpp_is_last_stage(), start, end


def _collect_aux_hidden_state(
    aux_hidden_states: list,
    hidden_states,
    residual,
) -> None:
    aux_hidden_states.append(
        hidden_states + residual if residual is not None else hidden_states
    )


_original_llama_model_init = LlamaModel.__init__
_original_llama_model_forward = LlamaModel.forward
_original_llama_causal_lm_init = LlamaForCausalLM.__init__
_original_llama_compute_logits = LlamaForCausalLM.compute_logits


def _vpp_llama_model_init(
    self,
    *,
    vllm_config,
    prefix: str = "",
    layer_type=LlamaDecoderLayer,
):
    vp_size = _get_vp_size()
    if vp_size <= 1 or self.__class__ is not LlamaModel:
        _original_llama_model_init(
            self,
            vllm_config=vllm_config,
            prefix=prefix,
            layer_type=layer_type,
        )
        return

    nn.Module.__init__(self)
    self.vllm_config = vllm_config
    self.compilation_config = vllm_config.compilation_config
    self.do_not_compile = True

    config = vllm_config.model_config.hf_config
    quant_config = vllm_config.quant_config

    self.config = config
    self.quant_config = quant_config
    self.vocab_size = config.vocab_size

    pp_group = get_pp_group()
    has_embed = _vpp_rank_has_embedding(
        pp_group.rank_in_group,
        pp_group.world_size,
        vp_size,
        tie_word_embeddings=config.tie_word_embeddings,
    )
    if has_embed:
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
        )
    else:
        self.embed_tokens = PPMissingLayer()

    custom_ranges = _get_custom_layer_ranges_for_rank()
    self.vpp_layer_ranges, self.layers = make_vpp_layers(
        config.num_hidden_layers,
        lambda layer_prefix: layer_type(
            vllm_config=vllm_config,
            prefix=layer_prefix,
        ),
        prefix=f"{prefix}.layers",
        vp_size=vp_size,
        custom_layer_ranges=custom_ranges,
    )
    self.start_layer = self.vpp_layer_ranges[0][0]
    self.end_layer = self.vpp_layer_ranges[-1][1]

    if _vpp_rank_has_output(
        pp_group.rank_in_group, pp_group.world_size, vp_size
    ):
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    else:
        self.norm = PPMissingLayer()

    self.aux_hidden_state_layers: tuple[int, ...] = ()
    self.make_empty_intermediate_tensors = (
        make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
    )


def _vpp_llama_model_forward(
    self,
    input_ids,
    positions,
    intermediate_tensors=None,
    inputs_embeds=None,
    **extra_layer_kwargs,
):
    if not hasattr(self, "vpp_layer_ranges"):
        return _original_llama_model_forward(
            self,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **extra_layer_kwargs,
        )

    is_first, is_last, start, end = _vpp_stage_bounds(self)

    if is_first:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    aux_hidden_states = []
    for layer_idx, layer in enumerate(
        islice(self.layers, start, end),
        start=start,
    ):
        if layer_idx in self.aux_hidden_state_layers:
            _collect_aux_hidden_state(aux_hidden_states, hidden_states, residual)
        hidden_states, residual = layer(
            positions,
            hidden_states,
            residual,
            **extra_layer_kwargs,
        )

    if not is_last:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )

    hidden_states, _ = self.norm(hidden_states, residual)
    if aux_hidden_states:
        return hidden_states, aux_hidden_states
    return hidden_states


def _vpp_llama_causal_lm_init(
    self,
    *,
    vllm_config,
    prefix: str = "",
    layer_type=LlamaDecoderLayer,
):
    vp_size = _get_vp_size()
    if vp_size <= 1 or self.__class__ is not LlamaForCausalLM:
        _original_llama_causal_lm_init(
            self,
            vllm_config=vllm_config,
            prefix=prefix,
            layer_type=layer_type,
        )
        self._llama_vpp_enabled = False
        return

    nn.Module.__init__(self)
    config = vllm_config.model_config.hf_config
    quant_config = vllm_config.quant_config

    self.config = config
    self.model = self._init_model(
        vllm_config=vllm_config,
        prefix=maybe_prefix(prefix, "model"),
        layer_type=layer_type,
    )

    pp_group = get_pp_group()
    if _vpp_rank_has_output(
        pp_group.rank_in_group, pp_group.world_size, vp_size
    ):
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
    else:
        self.lm_head = PPMissingLayer()

    logit_scale = getattr(config, "logit_scale", 1.0)
    self.logits_processor = LogitsProcessor(config.vocab_size, scale=logit_scale)
    self.make_empty_intermediate_tensors = (
        self.model.make_empty_intermediate_tensors
    )
    self._llama_vpp_enabled = True


def _vpp_llama_compute_logits(self, hidden_states):
    if not getattr(self, "_llama_vpp_enabled", False):
        return _original_llama_compute_logits(self, hidden_states)
    if isinstance(self.lm_head, PPMissingLayer):
        return None
    return self.logits_processor(self.lm_head, hidden_states)


LlamaModel.__init__ = _vpp_llama_model_init
LlamaModel.forward = _vpp_llama_model_forward
LlamaForCausalLM.__init__ = _vpp_llama_causal_lm_init
LlamaForCausalLM.compute_logits = _vpp_llama_compute_logits
