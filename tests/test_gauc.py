import pytest
from pathlib import Path
from auto_tune_weights_pipeline.events import Events
from auto_tune_weights_pipeline.target_config import TargetConfig
from auto_tune_weights_pipeline.metrics.gauc import GAUC
from auto_tune_weights_pipeline.columns import Columns


class TestGaucCalculator:
    @pytest.fixture
    def test_data_dir(self):
        return Path(__file__).parent.joinpath("data")

    @pytest.fixture
    def target_config_watch_coverage_30s(self):
        return {
            "watch_coverage_30s": TargetConfig(
                name="watch_coverage_30s",
                event_name=Events.WATCH_COVERAGE_RECORD,
                view_threshold_sec=30.0,
            )
        }

    @pytest.mark.auc
    def test_auc_calculating(
        self,
        test_data_dir,
        target_config_watch_coverage_30s,
    ):
        path_to_pool_cache = test_data_dir.joinpath(
            "./AUC=1_0/pool_cache_2026_01_31_rid=4209389434_2808_1769884974477_49306.jsonl"
        )
        target_event = "watch_coverage_30s"

        gauc_metric = GAUC(path_to_pool_cache=path_to_pool_cache)
        results = gauc_metric.calculate_metric(
            target_configs=target_config_watch_coverage_30s,
            session_col_name=Columns.RID_COL_NAME,
            nav_screen="video_for_you",
            platform="vk_video_android",
            calculate_regular_auc=True,
        )

        assert results[target_event] == {
            "AUC": 0.0,
            "AUC_error": None,
            "AUC_valid": True,
            "GAUCSimple": 1.0,
            "GAUCWeighted": 1.0,
            "GAUC_valid": True,
            "group_details": [
                {
                    "auc": 1.0,
                    "negatives": 1,
                    "positive_rate": 0.5,
                    "positives": 1,
                    "session_id": "4209389434.2808.1769884974477.49306",
                    "size": 2,
                },
            ],
            "max": 1.0,
            "median": 1.0,
            "min": 1.0,
            "n_groups": 1,
            "positive_rate": 0.5,
            "std": 0.0,
            "target": "watch_coverage_30s",
            "total_positives": 1,
            "total_samples": 2,
        }
