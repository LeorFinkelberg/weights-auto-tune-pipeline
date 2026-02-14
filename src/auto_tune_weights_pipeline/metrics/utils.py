import typing as t

from auto_tune_weights_pipeline.columns import Columns
from auto_tune_weights_pipeline.constants import NavScreens, Platforms, SummaryLogFields
from auto_tune_weights_pipeline.types_ import StrPath, TupleStrOrPlatforms
from auto_tune_weights_pipeline.target_config import TargetNames
from auto_tune_weights_pipeline.features_pairs_generator import FeaturesPairsGenerator
from auto_tune_weights_pipeline.ml import CatBoostPoolProcessor, PoolCacheInfo
from auto_tune_weights_pipeline.metrics.gauc import GAUC


def get_metric(
    trainer,
    target_config: dict,
    pool_cache_info_val: PoolCacheInfo,
    features_pairs_generator: FeaturesPairsGenerator,
    nav_screen: NavScreens = NavScreens.VIDEO_FOR_YOU,
    platforms: TupleStrOrPlatforms = (Platforms.ANDROID, Platforms.VK_VIDEO_ANDROID),
    target_details: SummaryLogFields = SummaryLogFields.TARGET_DETAILS,
    target_name: t.Union[str, TargetNames] = TargetNames.WATCH_COVERAGE_30S,
    metric_name: str = SummaryLogFields.GAUC_WEIGHTED,
    calculate_regular_auc: bool = True,
) -> float:
    _path_to_pool_cache_with_catboost_scores: StrPath = (
        CatBoostPoolProcessor.add_catboost_scores_to_pool_cache(
            trainer=trainer,
            pool_cache_info_val=pool_cache_info_val,
            features_pairs_generator=features_pairs_generator,
            score_col_name=Columns.CATBOOST_SCORE_COL_NAME,
        )
    )

    metric = GAUC(path_to_pool_cache=_path_to_pool_cache_with_catboost_scores)
    summary = GAUC.get_summary(
        metric.calculate_metric(
            target_configs=target_config,
            score_col_name=Columns.CATBOOST_SCORE_COL_NAME,
            session_col_name=Columns.RID_COL_NAME,
            nav_screen=nav_screen,
            platforms=platforms,
            calculate_regular_auc=calculate_regular_auc,
        )
    )

    return summary[target_details][target_name][metric_name]
