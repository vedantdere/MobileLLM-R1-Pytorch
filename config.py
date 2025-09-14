class PretrainedConfig:
    model_type=""
    base_config_key=""
    sub_configs={}
    has_no_defaults_at_init=None
    attribute_map={}
    base_model_tp_plan=None
    base_model_pp_plan=None
    
    def __init__(self,
                output_hidden_state=False,
                output_attentions=False,
                return_dict=True,
                torchscript=False,
                torch_dtype=False,
                purned_heads=None,
                tie_word_embeddings=True,
                chunk_size_feed_forward=0,
                is_encoder_decoder=False,
                is_decoder=False,
                cross_attention_hidden_size=None,
                and_cross_attention=False,
                tie_encoder_decoder=False,
                tokenizer_class=None,
                prefix=None,
                bos_token_id=None,
                pad_token_id=None,
                eos_token_id=None,
                sep_token_id=None,
                decoder_start_token_id=None
                ):
        self.return_dict=return_dict
        self.output_hidden_states=output_hidden_state
        self.torchscript=torchscript
        self.torch_dtype=torch_dtype
        self._output_attention-output_attentions

        self.pruned_heads=purned_heads if purned_heads is not None else {}
        self.tie_word_embeddings=tie_word_embeddings
        self.chunk_size_feed_forward=chunk_size_feed_forward
        self.is_encoder_decoder=is_encoder_decoder
        self.is_decoder=is_decoder
        self.cross_attention_hidden_size=cross_attention_hidden_size
        self.tie_encoder_decoder=tie_encoder_decoder


class MobileLLM_R1_360M(PretrainedConfig):
    def __init__(self):
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.attn_logit_softcapping = None
        self.bos_token_id = 128000
        self.eos_token_id = [
            128001,
            128008,
            128009
        ]
        self.attn_scale = 0.1
        self.attention_chunk_size = 32768
        self.attn_temperature_tuning = False
        self.final_logit_softcapping = None
        self.floor_scale = 8192
        self.for_llm_compressor = False
        self.head_dim = 64
        self.hidden_activation = "silu"
        self.hidden_size = 1024
        self.initializer_range = 0.02
        self.interleave_moe_layer_step = 0
        self.num_hidden_layers = 15
        self.intermediate_size = 4096
        self.intermediate_size_mlp = 4096
        self.layer_types = [
            "sliding_attention" if bool((i+1)%2) else "full_attention" for i in range(self.num_hidden_layers)
        ]

        self.max_position_embeddings = 32768
        self.model_type = "llama4_text"
        self.moe_layers = []
        self.no_rope_layers = [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1
        ]
        self.num_attention_heads = 16
        self.num_experts_per_tok = 0
        self.num_key_value_heads = 4
        self.num_local_experts = 0
        self.output_router_logits = False
        self.pad_token_id = 0
        self.query_pre_attn_scalar = 256
        self.rms_norm_eps = 1e-05
        self.rope_local_base_freq = 10000.0
        self.rope_scaling=None
        self.rope_theta = 8000000.0
        self.router_aux_loss_coef = 0.001
        self.router_jitter_noise = 0.0
        self.tie_word_embeddings = True
        self.sliding_window = 512
        self.use_cache = False
        self.output_hidden_states = None
        self.vocab_size = 128256
        torch_dtype = "float32"
        use_cache = True
        use_qk_norm = True
