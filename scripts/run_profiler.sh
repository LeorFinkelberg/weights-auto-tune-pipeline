uv run pyinstrument \
	-o auto_tune_profile.html \
	--color \
	cli.py \
		--path-to-pool-cache-train ./data/pool_cache_with_features_2026_02_01_train.jsonl \
		--path-to-pool-cache-val ./data/pool_cache_with_features_2026_02_02_val.jsonl \
		--loss-function PairLogitPairwise \
		--depth 3 \
		--timeout 300 \
		--n-trials 1 \
		--no-load-if-exists
