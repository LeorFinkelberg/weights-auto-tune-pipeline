import pytest

from pathlib import Path
from auto_tune_weights_pipeline.events import Events
from auto_tune_weights_pipeline.target_config import TargetConfig
from auto_tune_weights_pipeline.metrics.gauc import GAUC
from auto_tune_weights_pipeline.columns import Columns
from auto_tune_weights_pipeline.types_ import StrPath


def _get_rid_from_pool_cache_file_name(path_to_pool_cache_file: StrPath) -> str:
    return ".".join(
        str(Path(path_to_pool_cache_file).with_suffix("")).split("__")[-1].split("_")
    )


class TestGaucCalculator:
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
    def test_calculate__watch_coverage_30s__2records__auc_0_0(
        self,
        test_data_dir,
        target_config_watch_coverage_30s,
    ):
        # rid = 4209389434.2808.1769884974477.49306
        target_event = "watch_coverage_30s"
        path_to_pool_cache_file = test_data_dir.joinpath(
            "./metrics/gauc/AUC=0_0/pool_cache_2026_01_31__4209389434_2808_1769884974477_49306.jsonl"
        )
        rid = _get_rid_from_pool_cache_file_name(path_to_pool_cache_file)

        gauc_metric = GAUC(path_to_pool_cache=path_to_pool_cache_file)
        results = gauc_metric.calculate_metric(
            target_configs=target_config_watch_coverage_30s,
            session_col_name=Columns.RID_COL_NAME,
            nav_screen="video_for_you",
            platforms=("vk_video_android",),
            calculate_regular_auc=True,
        )

        assert results[target_event] == {
            "AUC": 0.0,
            "AUC_error": None,
            "AUC_valid": True,
            "GAUCSimple": 0.0,
            "GAUCWeighted": 0.0,
            "GAUC_valid": True,
            "group_details": [
                {
                    "auc": 0.0,
                    "negatives": 1,
                    "positive_rate": 0.5,
                    "positives": 1,
                    "session_id": rid,
                    "size": 2,
                },
            ],
            "max": 0.0,
            "median": 0.0,
            "min": 0.0,
            "n_groups": 1,
            "positive_rate": 0.5,
            "std": 0.0,
            "target": target_event,
            "total_positives": 1,
            "total_samples": 2,
        }

    @pytest.mark.auc
    @pytest.mark.parametrize(
        ["pool_cache_file"],
        [
            ("pool_cache_2026_01_31__918960451_4009_1769879722682_60050.jsonl",),
            ("pool_cache_2026_01_31__1972099784_3936_1769880273297_76459.jsonl",),
        ],
    )
    def test_calculate__watch_coverage_30s__3records__auc_0_0(
        self,
        pool_cache_file,
        test_data_dir,
        target_config_watch_coverage_30s,
    ):
        # rid =
        #     918960451.4009.1769879722682.60050
        #     1972099784.3936.1769880273297.76459
        target_event = "watch_coverage_30s"
        path_to_pool_cache_file = test_data_dir.joinpath(
            f"./metrics/gauc/AUC=0_0/{pool_cache_file}"
        )
        rid = _get_rid_from_pool_cache_file_name(path_to_pool_cache_file)

        gauc_metric = GAUC(path_to_pool_cache=path_to_pool_cache_file)
        results = gauc_metric.calculate_metric(
            target_configs=target_config_watch_coverage_30s,
            session_col_name=Columns.RID_COL_NAME,
            nav_screen="video_for_you",
            platforms=("vk_video_android",),
            calculate_regular_auc=True,
        )

        assert results[target_event] == {
            "AUC": 0.0,
            "AUC_error": None,
            "AUC_valid": True,
            "GAUCSimple": 0.0,
            "GAUCWeighted": 0.0,
            "GAUC_valid": True,
            "group_details": [
                {
                    "auc": 0.0,
                    "negatives": 2,
                    "positive_rate": 0.3333333333333333,
                    "positives": 1,
                    "session_id": rid,
                    "size": 3,
                },
            ],
            "max": 0.0,
            "median": 0.0,
            "min": 0.0,
            "n_groups": 1,
            "positive_rate": 0.3333333333333333,
            "std": 0.0,
            "target": target_event,
            "total_positives": 1,
            "total_samples": 3,
        }

    @pytest.mark.auc
    def test_calculate__watch_coverage_30s__5records__auc_0_0(
        self,
        test_data_dir,
        target_config_watch_coverage_30s,
    ):
        # rid = 3808953996.3934.1769877723930.44275
        target_event = "watch_coverage_30s"
        path_to_pool_cache_file = test_data_dir.joinpath(
            "./metrics/gauc/AUC=0_0/pool_cache_2026_01_31__3808953996_3934_1769877723930_44275.jsonl"
        )
        rid = _get_rid_from_pool_cache_file_name(path_to_pool_cache_file)

        gauc_metric = GAUC(path_to_pool_cache=path_to_pool_cache_file)
        results = gauc_metric.calculate_metric(
            target_configs=target_config_watch_coverage_30s,
            session_col_name=Columns.RID_COL_NAME,
            nav_screen="video_for_you",
            platforms=("vk_video_android",),
            calculate_regular_auc=True,
        )

        assert results[target_event] == {
            "AUC": 0.0,
            "AUC_error": None,
            "AUC_valid": True,
            "GAUCSimple": 0.0,
            "GAUCWeighted": 0.0,
            "GAUC_valid": True,
            "group_details": [
                {
                    "auc": 0.0,
                    "negatives": 3,
                    "positive_rate": 0.4,
                    "positives": 2,
                    "session_id": rid,
                    "size": 5,
                },
            ],
            "max": 0.0,
            "median": 0.0,
            "min": 0.0,
            "n_groups": 1,
            "positive_rate": 0.4,
            "std": 0.0,
            "target": target_event,
            "total_positives": 2,
            "total_samples": 5,
        }

    @pytest.mark.auc
    def test_calculate__watch_coverage_30s__3records__auc_0_5(
        self,
        test_data_dir,
        target_config_watch_coverage_30s,
    ):
        # rid = 3000127673.4130.1769889692846.13927
        target_event = "watch_coverage_30s"
        path_to_pool_cache_file = test_data_dir.joinpath(
            "./metrics/gauc/AUC=0_5/pool_cache_2026_01_31__3000127673_4130_1769889692846_13927.jsonl"
        )
        rid = _get_rid_from_pool_cache_file_name(path_to_pool_cache_file)

        gauc_metric = GAUC(path_to_pool_cache=path_to_pool_cache_file)
        results = gauc_metric.calculate_metric(
            target_configs=target_config_watch_coverage_30s,
            session_col_name=Columns.RID_COL_NAME,
            nav_screen="video_for_you",
            platforms=("vk_video_android",),
            calculate_regular_auc=True,
        )

        assert results[target_event] == {
            "AUC": 0.5,
            "AUC_error": None,
            "AUC_valid": True,
            "GAUCSimple": 0.5,
            "GAUCWeighted": 0.5,
            "GAUC_valid": True,
            "group_details": [
                {
                    "auc": 0.5,
                    "negatives": 2,
                    "positive_rate": 0.3333333333333333,
                    "positives": 1,
                    "session_id": rid,
                    "size": 3,
                },
            ],
            "max": 0.5,
            "median": 0.5,
            "min": 0.5,
            "n_groups": 1,
            "positive_rate": 0.3333333333333333,
            "std": 0.0,
            "target": target_event,
            "total_positives": 1,
            "total_samples": 3,
        }

    @pytest.mark.auc
    @pytest.mark.parametrize(
        ["pool_cache_file"],
        [
            ("pool_cache_2026_01_31__1550856805_3971_1769878063144_51518.jsonl",),
            ("pool_cache_2026_01_31__612205853_3532_1769849263613_98392.jsonl",),
        ],
    )
    def test_calculate__watch_coverage_30s__2records__auc_1_0(
        self,
        pool_cache_file,
        test_data_dir,
        target_config_watch_coverage_30s,
    ):
        # rid =
        #     1550856805.3971.1769878063144.51518
        #     612205853.3532.1769849263613.98392
        target_event = "watch_coverage_30s"
        path_to_pool_cache_file = test_data_dir.joinpath(
            f"./metrics/gauc/AUC=1_0/{pool_cache_file}"
        )
        rid = _get_rid_from_pool_cache_file_name(path_to_pool_cache_file)

        gauc_metric = GAUC(path_to_pool_cache=path_to_pool_cache_file)
        results = gauc_metric.calculate_metric(
            target_configs=target_config_watch_coverage_30s,
            session_col_name=Columns.RID_COL_NAME,
            nav_screen="video_for_you",
            platforms=("vk_video_android",),
            calculate_regular_auc=True,
        )

        assert results[target_event] == {
            "AUC": 1.0,
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
                    "session_id": rid,
                    "size": 2,
                },
            ],
            "max": 1.0,
            "median": 1.0,
            "min": 1.0,
            "n_groups": 1,
            "positive_rate": 0.5,
            "std": 0.0,
            "target": target_event,
            "total_positives": 1,
            "total_samples": 2,
        }
