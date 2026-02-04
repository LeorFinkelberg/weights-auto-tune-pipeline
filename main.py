from auto_tune_weights_pipeline.target_config import DEFAULT_TARGETS_CONFIG
from auto_tune_weights_pipeline.metrics.gauc import GAUC


# TODO: add @click
def main():
    result = GAUC(
        path_to_pool_cache="./data/pool_cache_2026_01_31.jsonl"
    ).calculate_metric(
        target_configs=DEFAULT_TARGETS_CONFIG,
        nav_screen="video_for_you",
        platform="vk_video_android",
    )

    print(result)


if __name__ == "__main__":
    main()
