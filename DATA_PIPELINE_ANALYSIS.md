# VERL Data Pipeline Analysis: From Parquet to Training Batch

## Executive Summary

This document traces how data flows from parquet files through verl's pipeline into the training batch's `non_tensor_batch`, specifically focusing on:
1. **`image_id`** propagation
2. **`reward_model.ground_truth`** (oracle graph JSON) propagation
3. **Available keys** in `batch.non_tensor_batch` during training

---

## Part 1: Dataset Stage (`RLHFDataset`)

### File: `verl/utils/dataset/rl_dataset.py`

**Source**: Parquet columns are:
- `data_source`
- `prompt`
- `image_id`
- `reward_model` (dict with `ground_truth` key)
- (other columns like images, videos, extra_info)

**Processing in `RLHFDataset.__getitem__()`** (lines 341-362):
```python
def __getitem__(self, item):
    row_dict: dict = self.dataframe[item]  # Load entire row from parquet
    row_dict["raw_prompt"] = self._build_messages(row_dict)
    
    # Add dummy tensor for DataProto.batch
    row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)
    
    # Extract extra_info fields
    row_dict["index"] = row_dict.get("extra_info", {}).get("index", 0)
    row_dict["tools_kwargs"] = row_dict.get("extra_info", {}).get("tools_kwargs", {})
    row_dict["interaction_kwargs"] = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
    
    return row_dict  # <-- Returns ALL fields from parquet, including image_id, reward_model, data_source, etc.
```

**Key insight**: The dataset's `__getitem__()` returns the **entire row** from the parquet file as a dictionary. This includes:
- ✅ `image_id` (if in parquet)
- ✅ `reward_model` dict (if in parquet)
- ✅ `data_source`
- ✅ `prompt`
- ✅ All other columns from parquet

---

## Part 2: Collation Stage (`collate_fn`)

### File: `verl/utils/dataset/rl_dataset.py` (lines 39-67)

```python
def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.
    
    - torch.Tensor values → stacked into (batch_size, *)
    - Non-tensor values → np.ndarray of dtype=object with shape (batch_size,)
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    
    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)
    
    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)
    
    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))
    
    return {**tensors, **non_tensors}
```

**Result**: A flat dictionary where:
- Tensor fields → `batch.batch[key]` (when wrapped in DataProto)
- Non-tensor fields → `batch.non_tensor_batch[key]` (when wrapped in DataProto)

**For parquet columns**:
- `image_id` → non-tensor → `batch.non_tensor_batch["image_id"]`
- `reward_model` (dict) → non-tensor → `batch.non_tensor_batch["reward_model"]`
- `data_source` (str) → non-tensor → `batch.non_tensor_batch["data_source"]`
- `prompt` → typically non-tensor → `batch.non_tensor_batch["prompt"]`

---

## Part 3: Training Loop - Batch Creation

### File: `verl/trainer/ppo/ray_trainer.py` (lines 1620-1642)

```python
# Line 1620: Load from dataloader (uses collate_fn)
for batch_dict in self.train_dataloader:
    metrics = {}
    timing_raw = {}
    
    # Line 1632: Wrap in DataProto
    batch: DataProto = DataProto.from_single_dict(batch_dict)
    
    # Line 1636-1638: Add UID (generated, not from parquet)
    batch.non_tensor_batch["uid"] = np.array(
        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
    )
    
    # Line 1640: Extract generation batch
    gen_batch = self._get_gen_batch(batch)
```

**At this point**, `batch.non_tensor_batch` contains:
- ✅ `image_id` (from parquet)
- ✅ `reward_model` (from parquet, includes `ground_truth`)
- ✅ `data_source` (from parquet)
- ✅ `prompt` (from parquet)
- ✅ `uid` (generated)
- ✅ `raw_prompt` (processed from dataset)
- ✅ `index` (from dataset)
- ✅ `tools_kwargs` (from dataset)
- ✅ `interaction_kwargs` (from dataset)
- ✅ And any other non-tensor columns from the parquet

---

## Part 4: Generation Batch Preparation

### File: `verl/trainer/ppo/ray_trainer.py` (lines 805-823)

```python
def _get_gen_batch(self, batch: DataProto) -> DataProto:
    # Keep these keys for generation
    reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid", "raw_prompt"}) \
                        & batch.non_tensor_batch.keys()
    
    # Pop all OTHER keys (to reduce memory)
    batch_keys_to_pop = []
    non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
    gen_batch = batch.pop(
        batch_keys=batch_keys_to_pop,
        non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
    )
    
    # For async rollout (agent loop), restore all keys
    if self.async_rollout_mode:
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
    else:
        if "raw_prompt" in batch.non_tensor_batch:
            gen_batch.non_tensor_batch["raw_prompt"] = batch.non_tensor_batch["raw_prompt"]
    
    return gen_batch
```

**Key insight**: Only these keys are sent to the rollout worker:
- `data_source`
- ✅ `reward_model` (preserved!)
- `extra_info`
- `uid`
- `raw_prompt`

**Note**: `image_id` is popped here (not in the reward_model_keys set) but remains in `batch` for later use.

---

## Part 5: Post-Rollout Union

### File: `verl/trainer/ppo/ray_trainer.py` (lines 1696-1698)

```python
# After rollout generation completes
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
batch = batch.union(gen_batch_output)
```

**After union**:
- Tensors from `gen_batch_output` are merged into `batch.batch`
- Non-tensors from original `batch` remain in `batch.non_tensor_batch`
- So `image_id`, `reward_model`, `data_source` are **still present**

---

## Part 6: Before Advantage Computation

### File: `verl/trainer/ppo/ray_trainer.py` (lines 1791-1844)

At line 1798, `_maybe_build_self_distillation_batch()` is called with:
- `batch: DataProto` (contains all the above)
- `reward_tensor`
- `reward_extra_infos_dict`

**Before advantage computation** (line 1836), `batch.non_tensor_batch` contains:
- ✅ `image_id`
- ✅ `reward_model` (with `ground_truth`)
- ✅ `data_source`
- ✅ `uid`
- ✅ `raw_prompt`
- ✅ `__num_turns__` (if from agent loop)
- + any reward_extra_infos fields added at line 1805

---

## Part 7: IsoGraph-Specific Extensions

### File: `verl/trainer/train_isograph_sdpo.py`

For IsoGraph SDPO, the trainer extends the flow:

1. **Oracle Graph Indexing** (lines 280-317):
   ```python
   # Load all oracle graphs from oracle_graph_dir into _oracle_graph_index
   # Keyed by image_id
   self._oracle_graph_index[image_id] = oracle_graph
   ```

2. **Oracle Graph Resolution** (lines 319-343):
   ```python
   def _resolve_oracle_graph(self, sample: Any) -> Optional[dict]:
       # Priority 1: image_id in sample → lookup in _oracle_graph_index
       if hasattr(sample, "image_id") and sample.image_id:
           if image_id in self._oracle_graph_index:
               return self._oracle_graph_index[image_id]
       
       # Priority 2: embedded oracle_graph in sample
       if hasattr(sample, "oracle_graph") and sample.oracle_graph:
           return sample.oracle_graph
       
       # Priority 3: single global oracle_graph_path
       ...
   ```

**Key insight**: The `image_id` from `batch.non_tensor_batch["image_id"]` is used to:
- Look up the oracle graph from the in-memory index
- Match against Member B's data-B/page_*.json files

---

## Summary: Available Keys in `batch.non_tensor_batch` During Training

### At Point: `_maybe_build_self_distillation_batch()` is called (line 1798)

| Key | Source | Type | Present? | Example |
|-----|--------|------|----------|---------|
| `image_id` | Parquet column | str/int | ✅ YES | `"page_1"`, `123` |
| `reward_model` | Parquet column (dict) | dict | ✅ YES | `{"ground_truth": {...}, "style": "rule"}` |
| `reward_model.ground_truth` | Parquet → `reward_model["ground_truth"]` | str/dict/JSON | ✅ YES | Oracle graph JSON string or dict |
| `data_source` | Parquet column | str | ✅ YES | `"member_b"` |
| `prompt` | Parquet column | list/str | ✅ YES | (depends on parquet structure) |
| `raw_prompt` | Dataset `_build_messages()` | list[dict] | ✅ YES | Processed chat messages |
| `uid` | Generated by trainer | str | ✅ YES | UUID string |
| `index` | Parquet `extra_info.index` | int | ✅ YES | Dataset sample index |
| `tools_kwargs` | Parquet `extra_info.tools_kwargs` | dict | ✅ MAYBE | Tool configuration |
| `interaction_kwargs` | Parquet `extra_info.interaction_kwargs` | dict | ✅ MAYBE | Interaction config |
| `__num_turns__` | AgentLoop output | int | ✅ YES (if from agent loop) | Multi-turn count |
| Reward extra info keys | `reward_fn()` return | varies | ✅ YES (if reward fn returns them) | Custom keys |
| `extra_info` | Parquet column | dict | ✅ YES (if not popped) | Any extra metadata |

### Accessibility Patterns:

**During training**:
```python
# Access from batch in fit() or other training methods:
image_id = batch.non_tensor_batch["image_id"]  # np.ndarray of shape (batch_size,)
oracle_graphs = batch.non_tensor_batch["reward_model"]  # np.ndarray of dict objects

# For single sample (e.g., during self-distillation):
for i in range(batch_size):
    image_id_i = batch.non_tensor_batch["image_id"][i]
    reward_model_i = batch.non_tensor_batch["reward_model"][i]
    ground_truth_i = reward_model_i["ground_truth"]
```

---

## Propagation Verification: IsoGraph Use Case

### Path for `image_id` and `oracle_graph`:

1. **Parquet** → column `image_id`, column `reward_model` (dict with `ground_truth`)
2. **Dataset** → `__getitem__()` returns both as-is
3. **Collate** → both become non-tensors
4. **Batch Creation** → in `batch.non_tensor_batch`
5. **Trainer** → accessible in `_maybe_build_self_distillation_batch()` and later
6. **IsoGraph** → can use `image_id` to look up oracle graphs from `_oracle_graph_index`

### Example: Using in Trainer

```python
def _maybe_build_self_distillation_batch(self, batch, reward_tensor, reward_extra_infos_dict):
    # Access image_id and oracle_graph
    image_ids = batch.non_tensor_batch.get("image_id")  # np.ndarray
    reward_models = batch.non_tensor_batch.get("reward_model")  # np.ndarray of dicts
    
    for i, image_id in enumerate(image_ids):
        oracle_graph = reward_models[i].get("ground_truth")
        # Use for DGR feedback generation, etc.
```

---

## Conclusion

✅ **`image_id`** flows through the entire pipeline and is accessible in `batch.non_tensor_batch` during training.

✅ **`reward_model.ground_truth`** (oracle graph JSON) flows through the entire pipeline as part of the `reward_model` dict in `batch.non_tensor_batch["reward_model"]`.

✅ **All available keys** in `batch.non_tensor_batch` during training include the above plus: `data_source`, `uid`, `raw_prompt`, `index`, `tools_kwargs`, `interaction_kwargs`, `__num_turns__`, and any custom reward extra info fields.

The data pipeline is **non-destructive** — parquet columns are preserved as non-tensor fields throughout the training loop.
