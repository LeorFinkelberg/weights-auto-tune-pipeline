from auto_tune_weights_pipeline.metrics.base import MetricNames
from auto_tune_weights_pipeline.pool_cache import PoolCache
from auto_tune_weights_pipeline.target_config import DEFAULT_TARGETS_CONFIG


def main():
    pool_cache = PoolCache(
        path_to_yt_table="//home/hc/ucp/vk_video/pool_caches/1d/2026-01-31",
        path_to_output="./pool_cache_test.jsonl",
        overwrite=True,
    )

    pool_cache.calculate_metric(
        metric=MetricNames.GAUC,
        target_configs=DEFAULT_TARGETS_CONFIG,
        nav_screen="VIDEO_SHOWCASE",
    )


if __name__ == "__main__":
    main()
