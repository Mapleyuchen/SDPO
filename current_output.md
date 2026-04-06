(IsoGraphTaskRunner pid=3047524) [validate_config] All configuration checks passed successfully!
(IsoGraphTaskRunner pid=3047524) /home/fanqi/SDPO/verl/trainer/train_isograph_sdpo.py:826: UserWarning: Disabled critic as algorithm.adv_estimator != gae. If it is not intended, please set critic.enable=True
(IsoGraphTaskRunner pid=3047524)   use_critic=need_critic(config),
(IsoGraphTaskRunner pid=3047524) /home/fanqi/SDPO/verl/utils/profiler/config.py:52: UserWarning: Torch profiler tool config is not fully supported now.
(IsoGraphTaskRunner pid=3047524)   warnings.warn("Torch profiler tool config is not fully supported now.", stacklevel=1)
(raylet) [2026-04-05 21:13:24,446 E 3046261 3046294] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2026-04-05_21-13-11_721984_3045524 is over 95% full, available space: 2.10976 GB; capacity: 467.89 GB. Object creation will fail if spilling is required.
(IsoGraphTaskRunner pid=3047524) Using dataset class: RLHFDataset
(IsoGraphTaskRunner pid=3047524) dataset len: 24
(IsoGraphTaskRunner pid=3047524) Setting TOKENIZERS_PARALLELISM=false for forked processes.
(IsoGraphTaskRunner pid=3047524) WARNING:2026-04-05 21:13:26,910:Setting TOKENIZERS_PARALLELISM=false for forked processes.
Filtering prompts longer than 1024 tokens (num_proc=1):   0%|          | 0/24 [00:00<?, ? examples/s]
Filtering prompts longer than 1024 tokens (num_proc=1): 100%|██████████| 24/24 [00:01<00:00, 18.55 examples/s]
(IsoGraphTaskRunner pid=3047524) filter dataset len: 24
(IsoGraphTaskRunner pid=3047524) Using dataset class: RLHFDataset
Filtering prompts longer than 1024 tokens (num_proc=1): 100%|██████████| 24/24 [00:01<00:00, 16.71 examples/s]
(IsoGraphTaskRunner pid=3047524) dataset len: 8
(IsoGraphTaskRunner pid=3047524) Setting TOKENIZERS_PARALLELISM=false for forked processes.
(IsoGraphTaskRunner pid=3047524) WARNING:2026-04-05 21:13:28,918:Setting TOKENIZERS_PARALLELISM=false for forked processes.
Filtering prompts longer than 1024 tokens (num_proc=1):   0%|          | 0/8 [00:00<?, ? examples/s]
Filtering prompts longer than 1024 tokens (num_proc=1): 100%|██████████| 8/8 [00:01<00:00,  7.56 examples/s]
(IsoGraphTaskRunner pid=3047524) filter dataset len: 8
(IsoGraphTaskRunner pid=3047524) [IsoGraph] Config: oracle_graph_path=/home/fanqi/SDPO/global_oracle_graph_demo.json, oracle_graph_dir=None, use_dummy_env=True
(IsoGraphTaskRunner pid=3047524) Size of train dataloader: 24, Size of val dataloader: 1
(IsoGraphTaskRunner pid=3047524) Total training steps: 72
(IsoGraphTaskRunner pid=3047524) [IsoGraph] DummyEnvironment initialized with oracle: /home/fanqi/SDPO/global_oracle_graph_demo.json
(IsoGraphTaskRunner pid=3047524) colocated worker base class <class 'verl.single_controller.base.worker.Worker'>
(IsoGraphTaskRunner pid=3047524) bind role actor_rollout_ref method chat_completion to class <class 'verl.single_controller.ray.base.create_colocated_worker_cls.<locals>.WorkerDict'>
(IsoGraphTaskRunner pid=3047524) bind role actor_rollout_ref method generate to class <class 'verl.single_controller.ray.base.create_colocated_worker_cls.<locals>.WorkerDict'>
(IsoGraphTaskRunner pid=3047524) bind role actor_rollout_ref method get_zeromq_address to class <class 'verl.single_controller.ray.base.create_colocated_worker_cls.<locals>.WorkerDict'>
(IsoGraphTaskRunner pid=3047524) bind role actor_rollout_ref method sleep to class <class 'verl.single_controller.ray.base.create_colocated_worker_cls.<locals>.WorkerDict'>
(IsoGraphTaskRunner pid=3047524) bind role actor_rollout_ref method wake_up to class <class 'verl.single_controller.ray.base.create_colocated_worker_cls.<locals>.WorkerDict'>
Filtering prompts longer than 1024 tokens (num_proc=1): 100%|██████████| 8/8 [00:01<00:00,  6.60 examples/s]
(IsoGraphTaskRunner pid=3047524) /home/fanqi/SDPO/verl/trainer/ppo/ray_trainer.py:361: UserWarning: Disabled critic as algorithm.adv_estimator != gae. If it is not intended, please set critic.enable=True
(IsoGraphTaskRunner pid=3047524)   self.use_critic = need_critic(self.config)
(raylet) [2026-04-05 21:13:34,465 E 3046261 3046294] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2026-04-05_21-13-11_721984_3045524 is over 95% full, available space: 2.10971 GB; capacity: 467.89 GB. Object creation will fail if spilling is required.
(pid=3048142) WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
(pid=3048142) I0000 00:00:1775394815.240912 3048142 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
(pid=3048142) I0000 00:00:1775394815.288760 3048142 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
(pid=3048142) To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
(pid=3048142) WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
(pid=3048142) I0000 00:00:1775394816.205424 3048142 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
(WorkerDict pid=3048142) [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
(WorkerDict pid=3048142) reference model: /home/fanqi/QwenVL25/Qwen/Qwen2.5-VL-7B-Instruct
(WorkerDict pid=3048142) The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. Note that this behavior will be extended to all models in a future release.
(WorkerDict pid=3048142) Model config after override: Qwen2_5_VLConfig {
(WorkerDict pid=3048142)   "architectures": [
(WorkerDict pid=3048142)     "Qwen2_5_VLForConditionalGeneration"
(WorkerDict pid=3048142)   ],
(WorkerDict pid=3048142)   "attention_dropout": 0.0,
(WorkerDict pid=3048142)   "attn_implementation": "sdpa",
(WorkerDict pid=3048142)   "bos_token_id": 151643,
(WorkerDict pid=3048142)   "dtype": "bfloat16",
(WorkerDict pid=3048142)   "eos_token_id": 151645,
(WorkerDict pid=3048142)   "hidden_act": "silu",
(WorkerDict pid=3048142)   "hidden_size": 3584,
(WorkerDict pid=3048142)   "image_token_id": 151655,
(WorkerDict pid=3048142)   "initializer_range": 0.02,
(WorkerDict pid=3048142)   "intermediate_size": 18944,
(WorkerDict pid=3048142)   "max_position_embeddings": 128000,
(WorkerDict pid=3048142)   "max_window_layers": 28,
(WorkerDict pid=3048142)   "model_type": "qwen2_5_vl",
(WorkerDict pid=3048142)   "num_attention_heads": 28,
(WorkerDict pid=3048142)   "num_hidden_layers": 28,
(WorkerDict pid=3048142)   "num_key_value_heads": 4,
(WorkerDict pid=3048142)   "rms_norm_eps": 1e-06,
(WorkerDict pid=3048142)   "rope_scaling": {
(WorkerDict pid=3048142)     "mrope_section": [
(WorkerDict pid=3048142)       16,
(WorkerDict pid=3048142)       24,
(WorkerDict pid=3048142)       24
(WorkerDict pid=3048142)     ],
(WorkerDict pid=3048142)     "rope_type": "default",
(WorkerDict pid=3048142)     "type": "default"
(WorkerDict pid=3048142)   },
(WorkerDict pid=3048142)   "rope_theta": 1000000.0,
(WorkerDict pid=3048142)   "sliding_window": 32768,
(WorkerDict pid=3048142)   "text_config": {
(WorkerDict pid=3048142)     "_name_or_path": "/home/fanqi/QwenVL25/Qwen/Qwen2.5-VL-7B-Instruct",
(WorkerDict pid=3048142)     "architectures": [
(WorkerDict pid=3048142)       "Qwen2_5_VLForConditionalGeneration"
(WorkerDict pid=3048142)     ],
(WorkerDict pid=3048142)     "attention_dropout": 0.0,
(WorkerDict pid=3048142)     "dtype": "bfloat16",
(WorkerDict pid=3048142)     "eos_token_id": 151645,
(WorkerDict pid=3048142)     "hidden_act": "silu",
(WorkerDict pid=3048142)     "hidden_size": 3584,
(WorkerDict pid=3048142)     "initializer_range": 0.02,
(WorkerDict pid=3048142)     "intermediate_size": 18944,
(WorkerDict pid=3048142)     "layer_types": [
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention"
(WorkerDict pid=3048142)     ],
(WorkerDict pid=3048142)     "max_position_embeddings": 128000,
(WorkerDict pid=3048142)     "max_window_layers": 28,
(WorkerDict pid=3048142)     "model_type": "qwen2_5_vl_text",
(WorkerDict pid=3048142)     "num_attention_heads": 28,
(WorkerDict pid=3048142)     "num_hidden_layers": 28,
(WorkerDict pid=3048142)     "num_key_value_heads": 4,
(WorkerDict pid=3048142)     "pad_token_id": 151643,
(WorkerDict pid=3048142)     "rms_norm_eps": 1e-06,
(WorkerDict pid=3048142)     "rope_scaling": {
(WorkerDict pid=3048142)       "mrope_section": [
(WorkerDict pid=3048142)         16,
(WorkerDict pid=3048142)         24,
(WorkerDict pid=3048142)         24
(WorkerDict pid=3048142)       ],
(WorkerDict pid=3048142)       "rope_type": "default",
(WorkerDict pid=3048142)       "type": "default"
(WorkerDict pid=3048142)     },
(WorkerDict pid=3048142)     "rope_theta": 1000000.0,
(WorkerDict pid=3048142)     "sliding_window": null,
(WorkerDict pid=3048142)     "use_cache": true,
(WorkerDict pid=3048142)     "use_sliding_window": false,
(WorkerDict pid=3048142)     "vision_token_id": 151654,
(WorkerDict pid=3048142)     "vocab_size": 152064
(WorkerDict pid=3048142)   },
(WorkerDict pid=3048142)   "tie_word_embeddings": false,
(WorkerDict pid=3048142)   "transformers_version": "4.57.1",
(WorkerDict pid=3048142)   "use_cache": true,
(WorkerDict pid=3048142)   "use_sliding_window": false,
(WorkerDict pid=3048142)   "video_token_id": 151656,
(WorkerDict pid=3048142)   "vision_config": {
(WorkerDict pid=3048142)     "depth": 32,
(WorkerDict pid=3048142)     "fullatt_block_indexes": [
(WorkerDict pid=3048142)       7,
(WorkerDict pid=3048142)       15,
(WorkerDict pid=3048142)       23,
(WorkerDict pid=3048142)       31
(WorkerDict pid=3048142)     ],
(WorkerDict pid=3048142)     "hidden_act": "silu",
(WorkerDict pid=3048142)     "hidden_size": 1280,
(WorkerDict pid=3048142)     "in_channels": 3,
(WorkerDict pid=3048142)     "in_chans": 3,
(WorkerDict pid=3048142)     "initializer_range": 0.02,
(WorkerDict pid=3048142)     "intermediate_size": 3420,
(WorkerDict pid=3048142)     "model_type": "qwen2_5_vl",
(WorkerDict pid=3048142)     "num_heads": 16,
(WorkerDict pid=3048142)     "out_hidden_size": 3584,
(WorkerDict pid=3048142)     "patch_size": 14,
(WorkerDict pid=3048142)     "spatial_merge_size": 2,
(WorkerDict pid=3048142)     "spatial_patch_size": 14,
(WorkerDict pid=3048142)     "temporal_patch_size": 2,
(WorkerDict pid=3048142)     "tokens_per_second": 2,
(WorkerDict pid=3048142)     "window_size": 112
(WorkerDict pid=3048142)   },
(WorkerDict pid=3048142)   "vision_end_token_id": 151653,
(WorkerDict pid=3048142)   "vision_start_token_id": 151652,
(WorkerDict pid=3048142)   "vision_token_id": 151654,
(WorkerDict pid=3048142)   "vocab_size": 152064
(WorkerDict pid=3048142) }
(WorkerDict pid=3048142) 
(WorkerDict pid=3048142) `torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards:   0%|          | 0/5 [00:00<?, ?it/s]
(raylet) [2026-04-05 21:13:44,488 E 3046261 3046294] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2026-04-05_21-13-11_721984_3045524 is over 95% full, available space: 2.1097 GB; capacity: 467.89 GB. Object creation will fail if spilling is required.
(raylet) [2026-04-05 21:13:54,512 E 3046261 3046294] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2026-04-05_21-13-11_721984_3045524 is over 95% full, available space: 2.1097 GB; capacity: 467.89 GB. Object creation will fail if spilling is required.
Loading checkpoint shards:  20%|██        | 1/5 [00:19<01:16, 19.17s/it]
(raylet) [2026-04-05 21:14:04,542 E 3046261 3046294] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2026-04-05_21-13-11_721984_3045524 is over 95% full, available space: 2.10969 GB; capacity: 467.89 GB. Object creation will fail if spilling is required.
(raylet) [2026-04-05 21:14:14,572 E 3046261 3046294] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2026-04-05_21-13-11_721984_3045524 is over 95% full, available space: 2.10966 GB; capacity: 467.89 GB. Object creation will fail if spilling is required.
Loading checkpoint shards:  40%|████      | 2/5 [00:41<01:03, 21.02s/it]
(raylet) [2026-04-05 21:14:24,596 E 3046261 3046294] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2026-04-05_21-13-11_721984_3045524 is over 95% full, available space: 2.10955 GB; capacity: 467.89 GB. Object creation will fail if spilling is required.
(raylet) [2026-04-05 21:14:34,623 E 3046261 3046294] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2026-04-05_21-13-11_721984_3045524 is over 95% full, available space: 2.10954 GB; capacity: 467.89 GB. Object creation will fail if spilling is required.
Loading checkpoint shards:  60%|██████    | 3/5 [01:04<00:43, 21.75s/it]
(raylet) [2026-04-05 21:14:44,645 E 3046261 3046294] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2026-04-05_21-13-11_721984_3045524 is over 95% full, available space: 2.10953 GB; capacity: 467.89 GB. Object creation will fail if spilling is required.
(raylet) [2026-04-05 21:14:54,664 E 3046261 3046294] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2026-04-05_21-13-11_721984_3045524 is over 95% full, available space: 2.10952 GB; capacity: 467.89 GB. Object creation will fail if spilling is required.
Loading checkpoint shards:  80%|████████  | 4/5 [01:22<00:20, 20.24s/it]
(raylet) [2026-04-05 21:15:04,689 E 3046261 3046294] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2026-04-05_21-13-11_721984_3045524 is over 95% full, available space: 2.10952 GB; capacity: 467.89 GB. Object creation will fail if spilling is required.
Loading checkpoint shards: 100%|██████████| 5/5 [01:32<00:00, 18.57s/it]
(WorkerDict pid=3048142) Monkey patch Qwen2_5_VLForConditionalGeneration model forward
(WorkerDict pid=3048142) Monkey patch Qwen2_5_VLForConditionalGeneration attention layer
(WorkerDict pid=3048142) Monkey patch _flash_attention_forward in transformers.integrations.flash_attention
(WorkerDict pid=3048142) Skipping monkey patch for Qwen2_5_VLForConditionalGeneration as use_fused_kernels is False or fused_kernels_backend is torch
(WorkerDict pid=3048142) Qwen2_5_VLForConditionalGeneration contains 8.29B parameters
(WorkerDict pid=3048142) wrap_policy: functools.partial(<function _or_policy at 0x7f680e5c82c0>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7f680e5c8180>, transformer_layer_cls={<class 'transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLVisionBlock'>, <class 'transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer'>})])
(WorkerDict pid=3048142) /home/fanqi/my_ai_env/EvaHan2026/lib/python3.12/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:479: UserWarning: FSDP is switching to use `NO_SHARD` instead of ShardingStrategy.FULL_SHARD since the world size is 1.
(WorkerDict pid=3048142)   _init_core_state(
(raylet) [2026-04-05 21:15:14,708 E 3046261 3046294] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2026-04-05_21-13-11_721984_3045524 is over 95% full, available space: 2.10949 GB; capacity: 467.89 GB. Object creation will fail if spilling is required.
(raylet) [2026-04-05 21:15:24,722 E 3046261 3046294] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2026-04-05_21-13-11_721984_3045524 is over 95% full, available space: 2.10932 GB; capacity: 467.89 GB. Object creation will fail if spilling is required.
(raylet) [2026-04-05 21:15:34,741 E 3046261 3046294] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2026-04-05_21-13-11_721984_3045524 is over 95% full, available space: 2.10931 GB; capacity: 467.89 GB. Object creation will fail if spilling is required.
(WorkerDict pid=3048142) Ref use_remove_padding=True
(WorkerDict pid=3048142) Ref use_fused_kernels=False
(WorkerDict pid=3048142) Ref use_prefix_grouper=False
(WorkerDict pid=3048142) Model config after override: Qwen2_5_VLConfig {
(WorkerDict pid=3048142)   "architectures": [
(WorkerDict pid=3048142)     "Qwen2_5_VLForConditionalGeneration"
(WorkerDict pid=3048142)   ],
(WorkerDict pid=3048142)   "attention_dropout": 0.0,
(WorkerDict pid=3048142)   "attn_implementation": "sdpa",
(WorkerDict pid=3048142)   "bos_token_id": 151643,
(WorkerDict pid=3048142)   "dtype": "bfloat16",
(WorkerDict pid=3048142)   "eos_token_id": 151645,
(WorkerDict pid=3048142)   "hidden_act": "silu",
(WorkerDict pid=3048142)   "hidden_size": 3584,
(WorkerDict pid=3048142)   "image_token_id": 151655,
(WorkerDict pid=3048142)   "initializer_range": 0.02,
(WorkerDict pid=3048142)   "intermediate_size": 18944,
(WorkerDict pid=3048142)   "max_position_embeddings": 128000,
(WorkerDict pid=3048142)   "max_window_layers": 28,
(WorkerDict pid=3048142)   "model_type": "qwen2_5_vl",
(WorkerDict pid=3048142)   "num_attention_heads": 28,
(WorkerDict pid=3048142)   "num_hidden_layers": 28,
(WorkerDict pid=3048142)   "num_key_value_heads": 4,
(WorkerDict pid=3048142)   "rms_norm_eps": 1e-06,
(WorkerDict pid=3048142)   "rope_scaling": {
(WorkerDict pid=3048142)     "mrope_section": [
(WorkerDict pid=3048142)       16,
(WorkerDict pid=3048142)       24,
(WorkerDict pid=3048142)       24
(WorkerDict pid=3048142)     ],
(WorkerDict pid=3048142)     "rope_type": "default",
(WorkerDict pid=3048142)     "type": "default"
(WorkerDict pid=3048142)   },
(WorkerDict pid=3048142)   "rope_theta": 1000000.0,
(WorkerDict pid=3048142)   "sliding_window": 32768,
(WorkerDict pid=3048142)   "text_config": {
(WorkerDict pid=3048142)     "_name_or_path": "/home/fanqi/QwenVL25/Qwen/Qwen2.5-VL-7B-Instruct",
(WorkerDict pid=3048142)     "architectures": [
(WorkerDict pid=3048142)       "Qwen2_5_VLForConditionalGeneration"
(WorkerDict pid=3048142)     ],
(WorkerDict pid=3048142)     "attention_dropout": 0.0,
(WorkerDict pid=3048142)     "dtype": "bfloat16",
(WorkerDict pid=3048142)     "eos_token_id": 151645,
(WorkerDict pid=3048142)     "hidden_act": "silu",
(WorkerDict pid=3048142)     "hidden_size": 3584,
(WorkerDict pid=3048142)     "initializer_range": 0.02,
(WorkerDict pid=3048142)     "intermediate_size": 18944,
(WorkerDict pid=3048142)     "layer_types": [
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention",
(WorkerDict pid=3048142)       "full_attention"
(WorkerDict pid=3048142)     ],
(WorkerDict pid=3048142)     "max_position_embeddings": 128000,
(WorkerDict pid=3048142)     "max_window_layers": 28,
(WorkerDict pid=3048142)     "model_type": "qwen2_5_vl_text",
(WorkerDict pid=3048142)     "num_attention_heads": 28,
(WorkerDict pid=3048142)     "num_hidden_layers": 28,
(WorkerDict pid=3048142)     "num_key_value_heads": 4,
(WorkerDict pid=3048142)     "pad_token_id": 151643,
(WorkerDict pid=3048142)     "rms_norm_eps": 1e-06,
(WorkerDict pid=3048142)     "rope_scaling": {
(WorkerDict pid=3048142)       "mrope_section": [
(WorkerDict pid=3048142)         16,
(WorkerDict pid=3048142)         24,
(WorkerDict pid=3048142)         24
(WorkerDict pid=3048142)       ],
(WorkerDict pid=3048142)       "rope_type": "default",
(WorkerDict pid=3048142)       "type": "default"
(WorkerDict pid=3048142)     },
(WorkerDict pid=3048142)     "rope_theta": 1000000.0,
(WorkerDict pid=3048142)     "sliding_window": null,
(WorkerDict pid=3048142)     "use_cache": true,
(WorkerDict pid=3048142)     "use_sliding_window": false,
(WorkerDict pid=3048142)     "vision_token_id": 151654,
(WorkerDict pid=3048142)     "vocab_size": 152064
(WorkerDict pid=3048142)   },
(WorkerDict pid=3048142)   "tie_word_embeddings": false,
(WorkerDict pid=3048142)   "transformers_version": "4.57.1",
(WorkerDict pid=3048142)   "use_cache": true,
(WorkerDict pid=3048142)   "use_sliding_window": false,
(WorkerDict pid=3048142)   "video_token_id": 151656,
(WorkerDict pid=3048142)   "vision_config": {
(WorkerDict pid=3048142)     "depth": 32,
(WorkerDict pid=3048142)     "fullatt_block_indexes": [
(WorkerDict pid=3048142)       7,
(WorkerDict pid=3048142)       15,
(WorkerDict pid=3048142)       23,
(WorkerDict pid=3048142)       31
(WorkerDict pid=3048142)     ],
(WorkerDict pid=3048142)     "hidden_act": "silu",
(WorkerDict pid=3048142)     "hidden_size": 1280,
(WorkerDict pid=3048142)     "in_channels": 3,
(WorkerDict pid=3048142)     "in_chans": 3,
(WorkerDict pid=3048142)     "initializer_range": 0.02,
(WorkerDict pid=3048142)     "intermediate_size": 3420,
(WorkerDict pid=3048142)     "model_type": "qwen2_5_vl",
(WorkerDict pid=3048142)     "num_heads": 16,
(WorkerDict pid=3048142)     "out_hidden_size": 3584,
(WorkerDict pid=3048142)     "patch_size": 14,
(WorkerDict pid=3048142)     "spatial_merge_size": 2,
(WorkerDict pid=3048142)     "spatial_patch_size": 14,
(WorkerDict pid=3048142)     "temporal_patch_size": 2,
(WorkerDict pid=3048142)     "tokens_per_second": 2,
(WorkerDict pid=3048142)     "window_size": 112
(WorkerDict pid=3048142)   },
(WorkerDict pid=3048142)   "vision_end_token_id": 151653,
(WorkerDict pid=3048142)   "vision_start_token_id": 151652,
(WorkerDict pid=3048142)   "vision_token_id": 151654,
(WorkerDict pid=3048142)   "vocab_size": 152064
(WorkerDict pid=3048142) }
(WorkerDict pid=3048142) 
Loading checkpoint shards:   0%|          | 0/5 [00:00<?, ?it/s]
(raylet) [2026-04-05 21:15:44,764 E 3046261 3046294] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2026-04-05_21-13-11_721984_3045524 is over 95% full, available space: 2.1093 GB; capacity: 467.89 GB. Object creation will fail if spilling is required.
(raylet) [2026-04-05 21:15:54,787 E 3046261 3046294] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2026-04-05_21-13-11_721984_3045524 is over 95% full, available space: 2.10929 GB; capacity: 467.89 GB. Object creation will fail if spilling is required.
Loading checkpoint shards:  20%|██        | 1/5 [00:19<01:19, 19.91s/it]
(raylet) [2026-04-05 21:16:04,811 E 3046261 3046294] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2026-04-05_21-13-11_721984_3045524 is over 95% full, available space: 2.10928 GB; capacity: 467.89 GB. Object creation will fail if spilling is required.
(raylet) [2026-04-05 21:16:14,834 E 3046261 3046294] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2026-04-05_21-13-11_721984_3045524 is over 95% full, available space: 2.10926 GB; capacity: 467.89 GB. Object creation will fail if spilling is required.
Error executing job with overrides: ['actor_rollout_ref.model.path=/home/fanqi/QwenVL25/Qwen/Qwen2.5-VL-7B-Instruct', '+actor_rollout_ref.model.override_config.attn_implementation=sdpa', 'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.rollout.n=4', 'trainer.total_epochs=3', 'trainer.n_gpus_per_node=1', 'trainer.nnodes=1', 'trainer.save_freq=10', 'trainer.test_freq=5', 'trainer.project_name=isograph_sdpo', 'trainer.experiment_name=qwen2_5_vl_7b_isograph', 'trainer.logger=console', 'data.train_files=/home/fanqi/SDPO/dummy_train.parquet', 'data.val_files=/home/fanqi/SDPO/dummy_val.parquet', 'isograph.use_dummy_env=true', 'isograph.svm_backend=dummy', 'trainer.device=cuda', 'actor_rollout_ref.rollout.name=hf', 'actor_rollout_ref.rollout.tensor_model_parallel_size=1', 'actor_rollout_ref.actor.use_torch_compile=false', 'actor_rollout_ref.ref.use_torch_compile=false', 'actor_rollout_ref.actor.fsdp_config.param_offload=true', 'actor_rollout_ref.actor.fsdp_config.optimizer_offload=true', 'actor_rollout_ref.ref.fsdp_config.param_offload=true', 'actor_rollout_ref.ref.fsdp_config.optimizer_offload=true', 'actor_rollout_ref.actor.policy_loss.isograph.ema_decay=0.99', 'actor_rollout_ref.actor.policy_loss.isograph.beta=0.01', 'actor_rollout_ref.actor.policy_loss.isograph.clip_ratio=0.2', 'isograph.oracle_graph_path=/home/fanqi/SDPO/global_oracle_graph_demo.json']
Traceback (most recent call last):
  File "/home/fanqi/SDPO/verl/trainer/train_isograph_sdpo.py", line 923, in main
    run_isograph_sdpo(config)
  File "/home/fanqi/SDPO/verl/trainer/train_isograph_sdpo.py", line 941, in run_isograph_sdpo
    ray.get(runner.run.remote(config))
  File "/home/fanqi/my_ai_env/EvaHan2026/lib/python3.12/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/fanqi/my_ai_env/EvaHan2026/lib/python3.12/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/fanqi/my_ai_env/EvaHan2026/lib/python3.12/site-packages/ray/_private/worker.py", line 2967, in get
    values, debugger_breakpoint = worker.get_objects(
                                  ^^^^^^^^^^^^^^^^^^^
  File "/home/fanqi/my_ai_env/EvaHan2026/lib/python3.12/site-packages/ray/_private/worker.py", line 1015, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(OutOfMemoryError): ray::IsoGraphTaskRunner.run() (pid=3047524, ip=192.168.1.103, actor_id=275d2b2aa3d81e2133388b0401000000, repr=<train_isograph_sdpo.IsoGraphTaskRunner object at 0x7fa12a7b8ec0>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/fanqi/SDPO/verl/trainer/train_isograph_sdpo.py", line 915, in run
    trainer.init_workers()
  File "/home/fanqi/SDPO/verl/trainer/ppo/ray_trainer.py", line 1160, in init_workers
    self.actor_rollout_wg.init_model()
  File "/home/fanqi/SDPO/verl/single_controller/ray/base.py", line 54, in __call__
    output = ray.get(output)
             ^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^
                                  ^^^^^^^^^^^^^^^^^^^
ray.exceptions.OutOfMemoryError: Task was killed due to the node running low on memory.
Memory on the node (IP: 192.168.1.103, ID: db89649fe40016559d1dd3384d3877a5614d60b8ea8b48068c636e98) where the lease (lease ID: 010000002e3a77fe4c5945aab2ea7157805027c656e12b9617fb36a5d6ebfc6b, name=1HsG1ZWorkerDict_0:0:WorkerDict.__init__, pid=3048142, memory used=49.21GB) was running was 59.84GB / 62.51GB (0.957347), which exceeds the memory usage threshold of 0.95. Ray killed this worker (ID: a4f59754f807e8590f26ce1ef9d39dbdd4220c57ddf2e6fe3c16df99) because it was the most recently scheduled task; to see more information about memory usage on this node, use `ray logs raylet.out -ip 192.168.1.103`. To see the logs of the worker, use `ray logs worker-a4f59754f807e8590f26ce1ef9d39dbdd4220c57ddf2e6fe3c16df99*out -ip 192.168.1.103. Top 10 memory users:
PID     MEM(GB) COMMAND
3048142 49.21   ray::WorkerDict.actor_rollout_ref_init_model
3047524 0.86    ray::IsoGraphTaskRunner.run
3045524 0.68    python -m verl.trainer.train_isograph_sdpo --config-name=isograph_sdpo actor_rollout_ref.model.path=...
3044269 0.50    /home/fanqi/.vscode-server/cli/servers/Stable-e7fb5e96c0730b9deb70b33781f98e2f35975036/server/node -...
3044860 0.41    /home/fanqi/.vscode-server/cli/servers/Stable-e7fb5e96c0730b9deb70b33781f98e2f35975036/server/node /...
3045622 0.14    /home/fanqi/my_ai_env/EvaHan2026/lib/python3.12/site-packages/ray/core/src/ray/gcs/gcs_server --log_...
3044281 0.10    /home/fanqi/.vscode-server/cli/servers/Stable-e7fb5e96c0730b9deb70b33781f98e2f35975036/server/node /...
3046345 0.09    /home/fanqi/my_ai_env/EvaHan2026/bin/python -u /home/fanqi/my_ai_env/EvaHan2026/lib/python3.12/site-...
3045714 0.07    /home/fanqi/my_ai_env/EvaHan2026/bin/python /home/fanqi/my_ai_env/EvaHan2026/lib/python3.12/site-pac...
3045793 0.07    ray-dashboard-ReportHead-0 (/home/fanqi/my_ai_env/EvaHan2026/bin/python -c "from multiprocessing.spa...
Refer to the documentation on how to address the out of memory issue: https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html. Consider provisioning more memory on this node or reducing task parallelism by requesting more CPUs per task. Set max_restarts and max_task_retries to enable retry when the task crashes due to OOM. To adjust the kill threshold, set the environment variable `RAY_memory_usage_threshold` when starting Ray. To disable worker killing, set the environment variable `RAY_memory_monitor_refresh_ms` to zero.

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
(/home/fanqi/my_ai_env/EvaHan2026) fanqi@oem-Precision-5820-Tower-X-Series:~/SDPO$ 