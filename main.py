from weights_auto_tune_pipeline.pool_cache import PoolCache
from weights_auto_tune_pipeline.target_config import DEFAULT_TARGETS_CONFIG


def main():
    pool_cache = PoolCache(
        path_to_yt_table="//home/hc/ucp/vk_video/pool_caches/1d/2026-01-31",
        path_to_output="./pool_cache_test.jsonl",
    )

    pool_cache.calculate_metric(
        target_configs=DEFAULT_TARGETS_CONFIG,
    )


if __name__ == "__main__":
    main()
