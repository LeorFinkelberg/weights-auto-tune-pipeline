uv run cli.py \
    --path-to-pool-cache-train ./data/pool_cache_with_features_2026_02_01_train.jsonl \
    --path-to-pool-cache-val ./data/pool_cache_with_features_2026_02_02_val.jsonl \
    --loss-function PairLogitPairwise \
    --timeout 3600 \
    --n-trials 50 \
    --no-load-if-exists
