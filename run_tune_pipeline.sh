uv run cli.py \
    --path-to-pool-cache-train ./data/pool_cache_with_features_2026_02_01_train.jsonl \
    --path-to-pool-cache-val ./data/pool_cache_with_features_2026_02_02_val.jsonl \
    --iterations 300 \
    --n-trials 5 \
    --timeout 600
