from mbridge.core import register_model
from mbridge.models import DeepseekV3Bridge
from megatron.core.transformer.enums import AttnBackend


@register_model("glm4_moe_lite")
class GLM4MoELiteBridge(DeepseekV3Bridge):
    def _build_config(self):
        hf_config = self.hf_config
        
        # Use getattr to safely access rope_theta with a default value
        rope_theta = getattr(hf_config, "rope_theta", 1000000)
        
        mla_rope_config = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "rope",
        }
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if rope_scaling is not None:
            mla_rope_config.update(rope_scaling)
        
        moe_layer_freq = [1] * hf_config.num_hidden_layers
        first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", 0)
        for i in range(min(first_k_dense_replace, hf_config.num_hidden_layers)):
            moe_layer_freq[i] = 0

        mtp_args = {}
        num_nextn_predict_layers = getattr(hf_config, "num_nextn_predict_layers", None)
        if num_nextn_predict_layers is not None:
            mtp_args["mtp_num_layers"] = num_nextn_predict_layers
            mtp_args["mtp_loss_scaling_factor"] = 0.1

        base_config = {
            "attention_backend": AttnBackend.fused,
            "layernorm_epsilon": hf_config.rms_norm_eps,
            "ffn_hidden_size": hf_config.intermediate_size,
            "qk_layernorm": True,
            # moe specific
            "moe_ffn_hidden_size": hf_config.moe_intermediate_size,
            "moe_token_dispatcher_type": "alltoall",
            "moe_router_bias_update_rate": 0.001,
            "moe_router_enable_expert_bias": True,
            "moe_router_topk": hf_config.num_experts_per_tok,
            "num_moe_experts": hf_config.n_routed_experts,
            "moe_shared_expert_intermediate_size": hf_config.moe_intermediate_size
            * getattr(hf_config, "n_shared_experts", 1),
            "moe_aux_loss_coeff": getattr(hf_config, "aux_loss_alpha", 0.001),
            "moe_router_load_balancing_type": "none",  # default None for RL
            "moe_shared_expert_overlap": True,
            "moe_grouped_gemm": True,
            "moe_router_score_function": "sigmoid",
            "moe_router_pre_softmax": True,
            "moe_router_topk_scaling_factor": getattr(hf_config, "routed_scaling_factor", 1.0),
            "moe_layer_freq": moe_layer_freq,
            # MLA
            "q_lora_rank": hf_config.q_lora_rank,
            "kv_lora_rank": hf_config.kv_lora_rank,
            "qk_head_dim": hf_config.qk_nope_head_dim,
            "qk_pos_emb_head_dim": hf_config.qk_rope_head_dim,
            "v_head_dim": hf_config.v_head_dim,
            "rotary_base": rope_theta,
            "rotary_scaling_factor": mla_rope_config["factor"],
            "rope_type": mla_rope_config["type"],
            "mscale": mla_rope_config["mscale"],
            "mscale_all_dim": mla_rope_config["mscale_all_dim"],
            "beta_fast": mla_rope_config["beta_fast"],
            "beta_slow": mla_rope_config["beta_slow"],
            # mcore 0.12 moe
            "moe_router_dtype": "fp32",
            "disable_bf16_reduced_precision_matmul": True,
            # other
            "persist_layer_norm": True,
            "bias_activation_fusion": True,
            "bias_dropout_fusion": True,
        }

        import megatron.core

        megatron_version = getattr(megatron.core, "__version__", "0.0")
        if megatron_version >= "0.14":
            base_config["original_max_position_embeddings"] = mla_rope_config[
                "original_max_position_embeddings"
            ]
        else:
            base_config["max_position_embeddings"] = mla_rope_config[
                "original_max_position_embeddings"
            ]

        base_config.update(mtp_args)
        return self._build_base_config(**base_config)

    def _get_gptmodel_args(self) -> dict:
        """
        Gets the arguments for GPTModel initialization.
        """
        rope_theta = getattr(self.hf_config, "rope_theta", 1000000)
        return dict(
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=rope_theta,
        )

