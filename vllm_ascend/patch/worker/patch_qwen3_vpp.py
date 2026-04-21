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
"""
VPP (Virtual Pipeline Parallelism) model patches for text-only Qwen3 models.

This patch intentionally targets only Qwen3ForCausalLM and
Qwen3MoeForCausalLM. Multimodal Qwen3 wrappers, Qwen3-Next, ASR, and Omni
models need separate model-specific VPP handling.
"""
from __future__ import annotations

from itertools import islice

from torch import nn
from vllm.distributed import get_ep_group, get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3 import (
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
)
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import is_interleaved

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


_original_qwen3_model_init = Qwen3Model.__init__
_original_qwen3_model_forward = Qwen3Model.forward
_original_qwen3_causal_lm_init = Qwen3ForCausalLM.__init__
_original_qwen3_compute_logits = Qwen3ForCausalLM.compute_logits
_original_qwen3_moe_model_init = Qwen3MoeModel.__init__
_original_qwen3_moe_model_forward = Qwen3MoeModel.forward
_original_qwen3_moe_causal_lm_init = Qwen3MoeForCausalLM.__init__
_original_qwen3_moe_compute_logits = Qwen3MoeForCausalLM.compute_logits
_original_qwen3_moe_update_experts = (
    Qwen3MoeForCausalLM.update_physical_experts_metadata
)


def _vpp_qwen3_model_init(self, *, vllm_config, prefix: str = ""):
    vp_size = _get_vp_size()
    if vp_size <= 1 or self.__class__ is not Qwen3Model:
        _original_qwen3_model_init(
            self, vllm_config=vllm_config, prefix=prefix)
        return

    nn.Module.__init__(self)
    self.vllm_config = vllm_config
    self.compilation_config = vllm_config.compilation_config
    self.do_not_compile = True

    config = vllm_config.model_config.hf_config.get_text_config()
    cache_config = vllm_config.cache_config
    quant_config = vllm_config.quant_config

    if is_interleaved(vllm_config.model_config.hf_text_config):
        assert config.max_window_layers == config.num_hidden_layers, (
            "Sliding window for some but all layers is not supported. "
            "This model uses sliding window but `max_window_layers` = {} "
            "is less than `num_hidden_layers` = {}. Please open an issue "
            "to discuss this feature.".format(
                config.max_window_layers,
                config.num_hidden_layers,
            )
        )

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
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
    else:
        self.embed_tokens = PPMissingLayer()

    custom_ranges = _get_custom_layer_ranges_for_rank()
    self.vpp_layer_ranges, self.layers = make_vpp_layers(
        config.num_hidden_layers,
        lambda prefix: Qwen3DecoderLayer(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        ),
        f"{prefix}.layers",
        vp_size,
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

    self.make_empty_intermediate_tensors = (
        make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
    )
    self.aux_hidden_state_layers: tuple[int, ...] = ()


def _vpp_qwen3_model_forward(
    self,
    input_ids,
    positions,
    intermediate_tensors=None,
    inputs_embeds=None,
):
    if not hasattr(self, "vpp_layer_ranges"):
        return _original_qwen3_model_forward(
            self, input_ids, positions, intermediate_tensors, inputs_embeds)

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
        hidden_states, residual = layer(positions, hidden_states, residual)

    if not is_last:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )

    hidden_states, _ = self.norm(hidden_states, residual)
    if aux_hidden_states:
        return hidden_states, aux_hidden_states
    return hidden_states


def _vpp_qwen3_causal_lm_init(self, *, vllm_config, prefix: str = ""):
    vp_size = _get_vp_size()
    if vp_size <= 1 or self.__class__ is not Qwen3ForCausalLM:
        _original_qwen3_causal_lm_init(
            self, vllm_config=vllm_config, prefix=prefix)
        self._qwen3_vpp_enabled = False
        return

    nn.Module.__init__(self)
    config = vllm_config.model_config.hf_config
    quant_config = vllm_config.quant_config

    self.config = config
    self.quant_config = quant_config
    self.model = Qwen3Model(
        vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
    )

    pp_group = get_pp_group()
    if _vpp_rank_has_output(
        pp_group.rank_in_group, pp_group.world_size, vp_size
    ):
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
    else:
        self.lm_head = PPMissingLayer()

    self.logits_processor = LogitsProcessor(config.vocab_size)
    self.make_empty_intermediate_tensors = (
        self.model.make_empty_intermediate_tensors
    )
    self._qwen3_vpp_enabled = True


def _vpp_qwen3_compute_logits(self, hidden_states):
    if not getattr(self, "_qwen3_vpp_enabled", False):
        return _original_qwen3_compute_logits(self, hidden_states)
    if isinstance(self.lm_head, PPMissingLayer):
        return None
    return self.logits_processor(self.lm_head, hidden_states)


def _vpp_qwen3_moe_model_init(
    self,
    *,
    vllm_config,
    prefix: str = "",
    decoder_layer_type=Qwen3MoeDecoderLayer,
):
    vp_size = _get_vp_size()
    if vp_size <= 1 or self.__class__ is not Qwen3MoeModel:
        _original_qwen3_moe_model_init(
            self,
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=decoder_layer_type,
        )
        return

    nn.Module.__init__(self)
    self.vllm_config = vllm_config
    self.compilation_config = vllm_config.compilation_config
    self.do_not_compile = True

    config = vllm_config.model_config.hf_text_config
    quant_config = vllm_config.quant_config
    eplb_config = vllm_config.parallel_config.eplb_config
    self.num_redundant_experts = eplb_config.num_redundant_experts

    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size
    self.config = config
    self.quant_config = quant_config

    pp_group = get_pp_group()
    has_embed = _vpp_rank_has_embedding(
        pp_group.rank_in_group,
        pp_group.world_size,
        vp_size,
        tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
    )
    if has_embed:
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
    else:
        self.embed_tokens = PPMissingLayer()

    custom_ranges = _get_custom_layer_ranges_for_rank()
    self.vpp_layer_ranges, self.layers = make_vpp_layers(
        config.num_hidden_layers,
        lambda prefix: decoder_layer_type(
            vllm_config=vllm_config, prefix=prefix),
        f"{prefix}.layers",
        vp_size,
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

    self.make_empty_intermediate_tensors = (
        make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
    )
    self.aux_hidden_state_layers: tuple[int, ...] = ()


def _vpp_qwen3_moe_model_forward(
    self,
    input_ids,
    positions,
    intermediate_tensors=None,
    inputs_embeds=None,
):
    if not hasattr(self, "vpp_layer_ranges"):
        return _original_qwen3_moe_model_forward(
            self, input_ids, positions, intermediate_tensors, inputs_embeds)

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
        hidden_states, residual = layer(positions, hidden_states, residual)

    if not is_last:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )

    hidden_states, _ = self.norm(hidden_states, residual)
    if aux_hidden_states:
        return hidden_states, aux_hidden_states
    return hidden_states


def _init_moe_metadata_from_local_layers(self, vllm_config) -> None:
    self.expert_weights = []
    self.moe_layers = []
    example_layer = None
    for layer in self.model.layers:
        if isinstance(layer, PPMissingLayer):
            continue
        assert isinstance(layer, Qwen3MoeDecoderLayer)
        if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
            example_layer = layer.mlp
            self.moe_layers.append(layer.mlp.experts)

    self.num_moe_layers = len(self.moe_layers)
    self.num_expert_groups = 1
    self.num_shared_experts = 0

    if example_layer is not None:
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts
        return

    config = vllm_config.model_config.hf_text_config
    num_redundant_experts = (
        vllm_config.parallel_config.eplb_config.num_redundant_experts
    )
    self.num_logical_experts = config.num_experts
    self.num_physical_experts = config.num_experts + num_redundant_experts
    self.num_local_physical_experts = (
        self.num_physical_experts // get_ep_group().world_size
    )
    self.num_routed_experts = config.num_experts
    self.num_redundant_experts = num_redundant_experts


def _vpp_qwen3_moe_causal_lm_init(
    self,
    *,
    vllm_config,
    prefix: str = "",
):
    vp_size = _get_vp_size()
    if vp_size <= 1 or self.__class__ is not Qwen3MoeForCausalLM:
        _original_qwen3_moe_causal_lm_init(
            self, vllm_config=vllm_config, prefix=prefix)
        self._qwen3_vpp_enabled = False
        return

    nn.Module.__init__(self)
    config = vllm_config.model_config.hf_text_config
    quant_config = vllm_config.quant_config
    self.config = config
    self.quant_config = quant_config
    if getattr(config, "mlp_only_layers", []):
        self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]

    self.model = Qwen3MoeModel(
        vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
    )

    pp_group = get_pp_group()
    if _vpp_rank_has_output(
        pp_group.rank_in_group, pp_group.world_size, vp_size
    ):
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
    else:
        self.lm_head = PPMissingLayer()

    self.logits_processor = LogitsProcessor(config.vocab_size)
    self.make_empty_intermediate_tensors = (
        self.model.make_empty_intermediate_tensors
    )
    _init_moe_metadata_from_local_layers(self, vllm_config)
    self._qwen3_vpp_enabled = True


def _vpp_qwen3_moe_compute_logits(self, hidden_states):
    if not getattr(self, "_qwen3_vpp_enabled", False):
        return _original_qwen3_moe_compute_logits(self, hidden_states)
    if isinstance(self.lm_head, PPMissingLayer):
        return None
    return self.logits_processor(self.lm_head, hidden_states)


def _vpp_qwen3_moe_update_physical_experts_metadata(
    self,
    num_physical_experts: int,
    num_local_physical_experts: int,
) -> None:
    if not getattr(self, "_qwen3_vpp_enabled", False):
        return _original_qwen3_moe_update_experts(
            self, num_physical_experts, num_local_physical_experts)

    assert self.num_local_physical_experts == num_local_physical_experts
    self.num_physical_experts = num_physical_experts
    self.num_local_physical_experts = num_local_physical_experts
    self.num_redundant_experts = num_physical_experts - self.num_logical_experts
    for layer in self.model.layers:
        if isinstance(layer, PPMissingLayer):
            continue
        assert isinstance(layer, Qwen3MoeDecoderLayer)
        if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
            moe = layer.mlp
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


Qwen3Model.__init__ = _vpp_qwen3_model_init
Qwen3Model.forward = _vpp_qwen3_model_forward
Qwen3ForCausalLM.__init__ = _vpp_qwen3_causal_lm_init
Qwen3ForCausalLM.compute_logits = _vpp_qwen3_compute_logits

Qwen3MoeModel.__init__ = _vpp_qwen3_moe_model_init
Qwen3MoeModel.forward = _vpp_qwen3_moe_model_forward
Qwen3MoeForCausalLM.__init__ = _vpp_qwen3_moe_causal_lm_init
Qwen3MoeForCausalLM.compute_logits = _vpp_qwen3_moe_compute_logits
Qwen3MoeForCausalLM.update_physical_experts_metadata = (
    _vpp_qwen3_moe_update_physical_experts_metadata
)
