import typing as t

from auto_tune_weights_pipeline.columns import Columns
from auto_tune_weights_pipeline.metrics.gauc import GAUC
from auto_tune_weights_pipeline.target_config import TargetConfig
from auto_tune_weights_pipeline.events import Events


# TODO: add @click
def main():
    target_config: t.Final[dict] = {
        "watch_coverage_30s": TargetConfig(
            name="watch_coverage_30s",
            event_name=Events.WATCH_COVERAGE_RECORD,
            view_threshold_sec=30.0,
        )
    }

    result = GAUC(
        path_to_pool_cache="./data/pool_cache_2026_01_31.jsonl"
    ).calculate_metric(
        target_configs=target_config,
        session_col_name=Columns.RID_COL_NAME,
        nav_screen="video_for_you",
        platform="vk_video_android",
    )

    print(result)


if __name__ == "__main__":
    main()
