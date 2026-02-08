import typing as t

import polars as pl
import catboost as cb

from loguru import logger

from auto_tune_weights_pipeline.types_ import StrPath
from auto_tune_weights_pipeline.columns import Columns
from auto_tune_weights_pipeline.metrics.gauc import GAUC
from auto_tune_weights_pipeline.target_config import TargetConfig
from auto_tune_weights_pipeline.events import Events
from auto_tune_weights_pipeline.logging_config import setup_logging
from auto_tune_weights_pipeline.features_pairs_generator import FeaturesPairsGenerator
from auto_tune_weights_pipeline.ml import CatboostTrainer, CatBoostPoolProcessor


setup_logging()


def main():
    PATH_TO_POOL_CACHE_TRAIN = "./data/pool_cache_with_features_2026_02_01_train.jsonl"
    PATH_TO_POOL_CACHE_VAL = "./data/pool_cache_with_features_2026_02_02_val.jsonl"

    features_pairs_generator = FeaturesPairsGenerator(
        path_to_feature_names="./feature_names.txt"
    )

    pool_cache_train = pl.read_ndjson(PATH_TO_POOL_CACHE_TRAIN)
    pool_cache_val = pl.read_ndjson(PATH_TO_POOL_CACHE_VAL)

    features_table_train = features_pairs_generator.generate_features_table(
        pool_cache_train
    )
    pairs_table_train = features_pairs_generator.generate_pairs_table(
        features_table_train
    )

    features_table_val = features_pairs_generator.generate_features_table(
        pool_cache_val
    )
    pairs_table_val = features_pairs_generator.generate_pairs_table(features_table_val)

    catboost_pool_processor = CatBoostPoolProcessor(
        features_table_train, pairs_table_train
    )
    pool_train: cb.Pool = catboost_pool_processor.create_pool()

    catboost_pool_processor = CatBoostPoolProcessor(features_table_val, pairs_table_val)
    pool_val: cb.Pool = catboost_pool_processor.create_pool()

    trainer = CatboostTrainer()
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
            path_to_pool_cache_val=PATH_TO_POOL_CACHE_VAL,
            pool_val=pool_val,
            features_val=features_table_val,
            score_col_name="catboost_score",
        )
    )

    metric = GAUC(path_to_pool_cache=path_to_pool_cache_with_catboost_scores)
    results = metric.calculate_metric(
        target_configs=target_config,
        session_col_name=Columns.RID_COL_NAME,
        nav_screen="video_for_you",
        platform="vk_video_android",
        calculate_regular_auc=True,
    )

    summary = GAUC.get_summary(results)
    logger.debug(summary)


if __name__ == "__main__":
    main()
