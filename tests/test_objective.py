import pytest
import typing as t
import polars as pl
import numpy as np
import catboost as cb

from unittest.mock import patch, Mock, MagicMock
from optuna.trial import FixedTrial

from auto_tune_weights_pipeline.constants import NavScreens, Platforms
from auto_tune_weights_pipeline.tune import Objective
from auto_tune_weights_pipeline.features_pairs_generator import FeaturesPairsGenerator
from auto_tune_weights_pipeline.ml import CatboostTrainer, PoolCacheInfo
from auto_tune_weights_pipeline.target_config import TargetConfig, TargetNames
from auto_tune_weights_pipeline.event_names import EventNames


class TestObjective:
    @pytest.mark.obj
    def test_objective_mock_training_only(self, test_data_dir, mock_dictionary_hub):
        target_config: t.Final[dict] = {
            TargetNames.WATCH_COVERAGE_30S: TargetConfig(
                target_name=TargetNames.WATCH_COVERAGE_30S,
                event_name=EventNames.WATCH_COVERAGE_RECORD,
                view_threshold_sec=30.0,
            )
        }

        path_to_pool_cache_train = test_data_dir.joinpath(
            "./objective/pool_cache_with_features_2026_02_01_train_5_records.jsonl"
        )
        path_to_pool_cache_val = test_data_dir.joinpath(
            "./objective/pool_cache_with_features_2026_02_02_val_3_records.jsonl"
        )
        pool_cache_info_train = PoolCacheInfo(
            data=pl.read_ndjson(path_to_pool_cache_train),
            path_to_data=path_to_pool_cache_train,
        )
        pool_cache_info_val = PoolCacheInfo(
            data=pl.read_ndjson(path_to_pool_cache_val),
            path_to_data=path_to_pool_cache_val,
        )
        path_to_feature_names = test_data_dir.joinpath("./feature_names.txt")

        features_pairs_generator = FeaturesPairsGenerator(
            path_to_feature_names=path_to_feature_names
        )

        mock_model = MagicMock(spec=cb.CatBoostRanker)
        mock_model.predict.return_value = np.array([0.1, 0.2, 0.4, 0.8, 0.1])

        mock_trainer_instance = MagicMock(spec=CatboostTrainer)
        mock_trainer_instance.ranker = mock_model
        mock_trainer_instance.train = Mock()

        mock_gauc_instance = MagicMock()
        mock_gauc_instance.calculate_metric.return_value = {
            "watch_coverage_30s": {
                "GAUCWeighted": 0.6572,
                "GAUCSimple": 0.5,
                "GAUC_valid": True,
                "AUC": 0.6,
                "AUC_valid": True,
            }
        }

        mock_add_scores = Mock(return_value="/tmp/test_scores.jsonl")

        with (
            patch(
                "auto_tune_weights_pipeline.tune.CatboostTrainer",
                return_value=mock_trainer_instance,
            ),
            patch(
                "auto_tune_weights_pipeline.metrics.utils.GAUC",
                return_value=mock_gauc_instance,
            ),
            patch.object(CatboostTrainer, "train"),
            patch(
                "auto_tune_weights_pipeline.ml.CatBoostPoolProcessor.add_catboost_scores_to_pool_cache",
                mock_add_scores,
            ),
            patch(
                "auto_tune_weights_pipeline.metrics.utils.GAUC.get_summary"
            ) as mock_get_summary,
        ):
            mock_get_summary.return_value = {
                "total_targets": 1,
                "targets_with_valid_auc": 1,
                "targets_with_valid_gauc": 1,
                "average_auc": 0.6,
                "average_gauc_weighted": 0.6572,
                "average_gauc_simple": 0.5,
                "target_details": {
                    "watch_coverage_30s": {
                        "GAUCWeighted": 0.6572,
                        "GAUCSimple": 0.5,
                        "GAUC_valid": True,
                        "AUC": 0.6,
                        "AUC_valid": True,
                        "n_groups": 10,
                        "total_samples": 100,
                        "positive_rate": 0.1,
                    }
                },
            }

            objective = Objective(
                target_config=target_config,
                pool_cache_info_train=pool_cache_info_train,
                pool_cache_info_val=pool_cache_info_val,
                features_pairs_generator=features_pairs_generator,
                catboost_params={
                    "iterations": 50,
                    "l2_leaf_reg": 3.0,
                },
                nav_screen=NavScreens.VIDEO_FOR_YOU,
                platforms=(Platforms.ANDROID, Platforms.VK_VIDEO_ANDROID),
            )

            trial = FixedTrial(
                {
                    "like_weight": 5.0,
                    "dislike_weight": 14.0,
                    "consumption_time_weight": 0.41,
                }
            )

            assert np.isclose(0.6572, objective(trial))
