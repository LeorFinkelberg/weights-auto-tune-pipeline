import typing as t

import numpy as np
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
        _add_catboost_scores_to_pool_cache(
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


def _add_catboost_scores_to_pool_cache(
    ranker: cb.CatBoostRanker,
    path_to_pool_cache_val: StrPath,
    pool_val: cb.Pool,
    features_val: pl.DataFrame,
    score_col_name: str = "catboost_score",
    output_path: t.Optional[StrPath] = None,
) -> StrPath:
    pool_cache_val = pl.read_ndjson(path_to_pool_cache_val)
    logger.info(f"Loaded pool cache: {len(pool_cache_val)} rows")
    logger.debug(f"Pool cache columns: {pool_cache_val.columns}")

    logger.info("Getting predictions from ranker ...")
    predictions: np.ndarray[float] = ranker.predict(pool_val)
    logger.debug(f"Predictions shape: {predictions.shape}")
    logger.debug(f"Features_val columns: {features_val.columns}")

    rid_column_in_features = None
    possible_rid_columns = ["original_rid", "rid", "session_id", "groupId"]

    for col in possible_rid_columns:
        if col in features_val.columns:
            rid_column_in_features = col
            logger.info(f"Found rid column in features_val: {rid_column_in_features}")
            break

    if rid_column_in_features is None:
        logger.error(
            f"No rid column found in features_val. Available columns: {features_val.columns}"
        )
        logger.info(f"First row of features_val: {features_val.row(0)}")
        raise ValueError("No rid column found in features_val")

    rid_column_in_pool_cache = None
    for col in possible_rid_columns:
        if col in pool_cache_val.columns:
            rid_column_in_pool_cache = col
            logger.info(
                f"Found rid column in pool_cache_val: {rid_column_in_pool_cache}"
            )
            break

    if rid_column_in_pool_cache is None:
        logger.error(
            f"No rid column found in pool_cache_val. Available columns: {pool_cache_val.columns}"
        )
        raise ValueError("No rid column found in pool_cache_val")

    scores_df = pl.DataFrame(
        {
            rid_column_in_pool_cache: features_val[rid_column_in_features],
            score_col_name: predictions,
        }
    )

    logger.info(f"Scores DataFrame: {len(scores_df)} rows")
    logger.info(
        f"Unique {rid_column_in_pool_cache} in scores: {scores_df[rid_column_in_pool_cache].n_unique()}"
    )
    logger.info(
        f"Sample rids from scores_df: {scores_df[rid_column_in_pool_cache].head(5).to_list()}"
    )

    scores_unique = scores_df[rid_column_in_pool_cache].n_unique()
    if scores_unique != len(scores_df):
        logger.warning(
            f"Duplicate keys in scores_df: {scores_unique} unique out of {len(scores_df)}"
        )
        duplicate_counts = (
            scores_df.group_by(rid_column_in_pool_cache)
            .agg(pl.len().alias("count"))
            .filter(pl.col("count") > 1)
        )
        if duplicate_counts.height > 0:
            logger.debug(f"Duplicate rids: {duplicate_counts.head(3)}")

        scores_df = scores_df.unique(subset=[rid_column_in_pool_cache], keep="last")
        logger.info(f"After deduplication: {len(scores_df)} rows")

    logger.info(f"Joining using key column: {rid_column_in_pool_cache}")

    pool_cache_with_scores = pool_cache_val.join(
        scores_df, on=rid_column_in_pool_cache, how="left"
    )

    null_count = pool_cache_with_scores[score_col_name].null_count()
    total_rows = len(pool_cache_with_scores)

    logger.info(
        f"Result: {total_rows - null_count} rows with scores, {null_count} rows without"
    )
    logger.info(f"Success rate: {(total_rows - null_count) / total_rows * 100:.1f}%")

    if 0 < null_count < 10:
        missing = (
            pool_cache_with_scores.filter(pl.col(score_col_name).is_null())
            .select([rid_column_in_pool_cache])
            .head(5)
        )
        logger.debug(
            f"Sample rids without scores: {missing[rid_column_in_pool_cache].to_list()}"
        )

    if output_path is None:
        output_path = path_to_pool_cache_val.replace(".jsonl", "_with_scores.jsonl")

    logger.info(f"Saving to {output_path}")
    pool_cache_with_scores.write_ndjson(output_path)

    loaded_check = pl.read_ndjson(output_path)
    logger.info(f"Saved file check: {len(loaded_check)} rows")

    if score_col_name in loaded_check.columns:
        loaded_null_count = loaded_check[score_col_name].null_count()
        logger.warning(f"Null scores in saved file: {loaded_null_count}")

        if loaded_null_count < total_rows:
            valid_scores = loaded_check[score_col_name].drop_nulls()
            logger.info(
                f"Score stats - min: {valid_scores.min():.4f}, "
                f"max: {valid_scores.max():.4f}, "
                f"mean: {valid_scores.mean():.4f}"
            )
    else:
        logger.error(f"Score column {score_col_name} not found in saved file!")

    return output_path


if __name__ == "__main__":
    main()
