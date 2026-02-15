import typing as t
import catboost as cb

from optuna.trial._trial import Trial

from auto_tune_weights_pipeline.features_pairs_generator import FeaturesPairsGenerator
from auto_tune_weights_pipeline.types_ import TupleStrOrPlatforms
from auto_tune_weights_pipeline.ml import (
    CatboostTrainer,
    CatBoostPoolProcessor,
    PoolCacheInfo,
)
from auto_tune_weights_pipeline.target_config import TargetNames
from auto_tune_weights_pipeline.constants import SummaryLogFields, Platforms, NavScreens
from auto_tune_weights_pipeline.metrics.utils import get_metric
from auto_tune_weights_pipeline.utils import Timer


class Objective:
    def __init__(
        self,
        target_config: dict,
        pool_cache_info_train: PoolCacheInfo,
        pool_cache_info_val: PoolCacheInfo,
        features_pairs_generator: FeaturesPairsGenerator,
        catboost_params: dict,
        nav_screen: t.Union[str, NavScreens] = NavScreens.VIDEO_FOR_YOU,
        platforms: TupleStrOrPlatforms = (Platforms.VK_VIDEO_ANDROID,),
        target_details=SummaryLogFields.TARGET_DETAILS,
        target_name=TargetNames.WATCH_COVERAGE_30S,
        metric_name=SummaryLogFields.GAUC_WEIGHTED,
        calculate_regular_auc=True,
    ) -> None:
        self.target_config = target_config
        self.pool_cache_info_train = pool_cache_info_train
        self.pool_cache_info_val = pool_cache_info_val
        self.features_pairs_generator = features_pairs_generator
        self.nav_screen = nav_screen
        self.platforms = platforms
        self.target_details = target_details
        self.target_name = target_name
        self.metric_name = metric_name
        self.calculate_regular_auc = calculate_regular_auc
        self.catboost_params = catboost_params

    def __call__(self, trial: Trial) -> float:
        like_weight = trial.suggest_float("like_weight", 0.0, 1_000.0)
        dislike_weight = trial.suggest_float("dislike_weight", 0.0, 1_000.0)
        consumption_time_weight = trial.suggest_float(
            "consumption_time_weight", 0.0, 100.0
        )

        with Timer.get_block_time("features and pairs"):
            features_table_train = (
                self.features_pairs_generator.generate_features_table(
                    self.pool_cache_info_train.data,
                    like_weight=like_weight,
                    dislike_weight=dislike_weight,
                    consumption_time_weight=consumption_time_weight,
                )
            )
            pairs_table_train = self.features_pairs_generator.generate_pairs_table(
                features_table_train
            )

        pool_train: cb.Pool = CatBoostPoolProcessor(
            features_table_train, pairs_table_train
        ).create_pool()

        trainer = CatboostTrainer(self.catboost_params)
        trainer.train(pool_train)

        return get_metric(
            trainer=trainer,
            target_config=self.target_config,
            pool_cache_info_val=self.pool_cache_info_val,
            features_pairs_generator=self.features_pairs_generator,
            nav_screen=self.nav_screen,
            platforms=self.platforms,
            target_details=self.target_details,
            target_name=self.target_name,
            metric_name=self.metric_name,
            calculate_regular_auc=self.calculate_regular_auc,
        )
