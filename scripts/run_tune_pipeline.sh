uv run cli.py \
    --path-to-pool-cache-train ./data/pool_cache_with_features_2026_02_01_train.jsonl \
    --path-to-pool-cache-val ./data/pool_cache_with_features_2026_02_02_val.jsonl \
    --formula-path fstorage:vk_video_266_1769078359_f \
    --loss-function PairLogitPairwise \
    --depth 3 \
    --timeout 3600 \
    --n-trials 500 \
    --no-load-if-exists \
    --save-predictions
