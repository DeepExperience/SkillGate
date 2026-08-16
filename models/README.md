# External model restoration point

Model shards are intentionally excluded from the repository.

Restore or mount complete Hugging Face model directories here:

- Qwen3.5-9B/
- Qwen3.5-27B/
- Qwen3-Embedding-8B/
- Qwen3-Reranker-8B/

Run ../skillrl recipes and ../skillrl doctor after restoring assets.

Merged SFT models remain under GeneralAgent/sft_training/merged_models/. RL
exports belong to their owner at
experiments/rl/runs/<experiment_id>/model/exports/<export_id>/.
