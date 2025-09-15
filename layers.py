import math
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from attn_implementation import eager_paged_attention_forward
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask, create_chunked_causal_mask




class Llama4TextExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(self.gate_up_proj.shape[0], -1, self.hidden_size)
        gate_up = torch.bmm(hidden_states, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
        next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)
        next_states = next_states.view(-1, self.hidden_size)
        return next_states


# Phi3MLP
class Llama4TextMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = config.intermediate_size

        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.activation_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(down_proj)


class Llama4TextL2Norm(torch.nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x)

    def extra_repr(self):
        return f"eps={self.eps}"


class Llama4TextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """
        Llama4RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Llama4Router(nn.Linear):
    def __init__(self, config):
        super().__init__(config.hidden_size, config.num_local_experts, bias=False)
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

    def forward(self, hidden_states):
        router_logits = super().forward(hidden_states)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)
        router_scores = torch.full_like(router_logits, float("-inf")).scatter_(1, router_indices, router_top_value)
        router_scores = torch.nn.functional.sigmoid(router_scores.float()).to(router_scores.dtype)
        return router_scores, router_logits


class Llama4TextMoe(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.experts = Llama4TextExperts(config)
        self.router = Llama4Router(config)
        self.shared_expert = Llama4TextMLP(config)

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_scores, router_logits = self.router(hidden_states)
        routed_in = hidden_states.repeat(router_scores.shape[1], 1)
        routed_in = routed_in * router_scores.transpose(0, 1).reshape(-1, 1)
        routed_out = self.experts(routed_in)
        out = self.shared_expert(hidden_states)
        out.add_(routed_out.reshape(router_scores.shape[1], -1, routed_out.shape[-1]).sum(dim=0))
        return out, router_logits


class Llama4TextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        self.rope_type = "llama3" if config.rope_scaling is not None else "default"

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Convert to complex representation
            freqs_cis = freqs_cis * self.attention_scaling

        return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis[:, :, None, :]).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis[:, :, None, :]).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Adapted from transformers.models.llama.modeling_llama.eager_attention_forward -> llama4 doesn't cast attn weights to fp32
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def vision_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * module.head_dim**-0.5
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Llama4TextAttention(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attn_scale = config.attn_scale
        self.floor_scale = config.floor_scale
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.use_rope = config.no_rope_layers[layer_idx]
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        if self.config.use_qk_norm and self.use_rope:
            self.qk_norm = Llama4TextL2Norm(config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values,
        cache_position,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(*input_shape, -1, self.head_dim)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.use_rope:  
            query_states, key_states = apply_rotary_emb(
                query_states, key_states, position_embeddings.to(query_states.device)
            )

        if hasattr(self, "qk_norm"):
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = (
                torch.log(torch.floor((cache_position.float() + 1.0) / self.floor_scale) + 1.0) * self.attn_scale + 1.0
            )
            attn_scales = attn_scales.view((1, input_shape[-1], 1, 1)).expand((*input_shape, 1, 1))  # batch size > 1
            query_states = (query_states * attn_scales).to(query_states.dtype)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        attention_interface = eager_paged_attention_forward
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Llama4TextDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Llama4TextAttention(config, layer_idx)
        self.is_moe_layer = layer_idx in config.moe_layers
        if self.is_moe_layer:  # the 128E model interleaves dense / sparse
            self.feed_forward = Llama4TextMoe(config)
        else:
            self.feed_forward = Llama4TextMLP(config, intermediate_size=config.intermediate_size_mlp)

        self.input_layernorm = Llama4TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Llama4TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        cache_position,
        position_embeddings,
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attention_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + attention_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        if self.is_moe_layer:
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states.view(residual.shape)
        return hidden_states


class Llama4PreTrainedModel(nn.Module):
    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, Llama4TextRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, Llama4TextExperts):
            module.gate_up_proj.data.normal_(mean=0.0, std=std)
            module.down_proj.data.normal_(mean=0.0, std=std)
        elif isinstance(module, Llama4VisionModel):
            module.class_embedding.data.normal_(std=module.scale)
            module.positional_embedding_vlm.data.normal_(std=module.scale)


class Llama4TextModel(Llama4PreTrainedModel):

    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Llama4TextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Llama4TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Llama4TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        # self.post_init()

    
    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        cache_position,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device))

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "chunked_attention": create_chunked_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        freq_cis = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=freq_cis,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Llama4ForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model = Llama4TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        # self.post_init()


    def forward(
        self,
        input_ids = None,
        attention_mask= None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        cache_position = None,
        logits_to_keep = 0,
        **kwargs,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Llama4ForCausalLM

        >>> model = Llama4ForCausalLM.from_pretrained("meta-llama4/Llama4-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama4/Llama4-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return logits

