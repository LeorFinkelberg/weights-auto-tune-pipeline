import typing as t

import polars as pl
import catboost as cb

from auto_tune_weights_pipeline.features_pairs_generator import FeaturesPairsGenerator
from auto_tune_weights_pipeline.types_ import StrPath
from auto_tune_weights_pipeline.columns import Columns
from auto_tune_weights_pipeline.metrics.gauc import GAUC
from auto_tune_weights_pipeline.events import Events
from auto_tune_weights_pipeline.ml import CatboostTrainer, CatBoostPoolProcessor
from auto_tune_weights_pipeline.target_config import TargetConfig
from auto_tune_weights_pipeline.constants import SummaryLogFields


class Objective:
    def __init__(
        self,
        path_to_pool_cache_train: StrPath,
        path_to_pool_cache_val: StrPath,
        features_pairs_generator: FeaturesPairsGenerator,
        catboost_params: dict,
        nav_screen: str = "video_for_you",
        platform: str = "vk_video_android",
        calculate_regular_auc=True,
    ) -> None:
        self.path_to_pool_cache_train = path_to_pool_cache_train
        self.path_to_pool_cache_val = path_to_pool_cache_val
        self.features_pairs_generator = features_pairs_generator
        self.nav_screen = nav_screen
        self.platform = platform
        self.calculate_regular_auc = calculate_regular_auc
        self.catboost_params = catboost_params

    def __call__(self, trial) -> float:
        pool_cache_train = pl.read_ndjson(self.path_to_pool_cache_train)
        pool_cache_val = pl.read_ndjson(self.path_to_pool_cache_val)

        like_weight = trial.suggest_float("like_weight", 0.0, 1_000)
        dislike_weight = trial.suggest_float("dislike_weight", 0.0, 1_000)
        consumption_time_weight = trial.suggest_float(
            "consumption_time_weight", 0.0, 10
        )

        features_table_train = self.features_pairs_generator.generate_features_table(
            pool_cache_train,
            like_weight=like_weight,
            dislike_weight=dislike_weight,
            consumption_time_weight=consumption_time_weight,
        )
        pairs_table_train = self.features_pairs_generator.generate_pairs_table(
            features_table_train
        )

        features_table_val = self.features_pairs_generator.generate_features_table(
            pool_cache_val,
            like_weight=like_weight,
            dislike_weight=dislike_weight,
            consumption_time_weight=consumption_time_weight,
        )
        pairs_table_val = self.features_pairs_generator.generate_pairs_table(
            features_table_val
        )

        catboost_pool_processor = CatBoostPoolProcessor(
            features_table_train, pairs_table_train
        )
        pool_train: cb.Pool = catboost_pool_processor.create_pool()

        catboost_pool_processor = CatBoostPoolProcessor(
            features_table_val, pairs_table_val
        )
        pool_val: cb.Pool = catboost_pool_processor.create_pool()

        trainer = CatboostTrainer(**self.catboost_params)
        trainer.train(pool_train)

        target_config: t.Final[dict] = {
            "watch_coverage_30s": TargetConfig(
                name="watch_coverage_30s",
                event_name=Events.WATCH_COVERAGE_RECORD,
                view_threshold_sec=30.0,
            )
        }

        path_to_pool_cache_with_catboost_scores: StrPath = (
            CatBoostPoolProcessor.add_catboost_scores_to_pool_cache(
                ranker=trainer.ranker,
                path_to_pool_cache_val=self.path_to_pool_cache_val,
                pool_val=pool_val,
                features_val=features_table_val,
                score_col_name="catboost_score",
            )
        )

        metric = GAUC(path_to_pool_cache=path_to_pool_cache_with_catboost_scores)
        results = metric.calculate_metric(
            target_configs=target_config,
            session_col_name=Columns.RID_COL_NAME,
            nav_screen=self.nav_screen,
            platform=self.platform,
            calculate_regular_auc=self.calculate_regular_auc,
        )

        summary = GAUC.get_summary(results)
        return summary["target_details"]["watch_coverage_30s"][
            SummaryLogFields.GAUC_WEIGHTED
        ]
