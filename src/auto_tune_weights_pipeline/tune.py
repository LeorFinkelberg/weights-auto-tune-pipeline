import typing as t
import catboost as cb
import numpy as np

from pathlib import Path
from loguru import logger

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


class Objective:
    def __init__(
        self,
        target_config: dict,
        pool_cache_info_train: PoolCacheInfo,
        pool_cache_info_val: PoolCacheInfo,
        features_pairs_generator: FeaturesPairsGenerator,
        catboost_params: dict,
        formula_path: str,
        session_col_name: str,
        alpha: float = 1.0,
        save_predictions: bool = False,
        path_to_baseline_model: t.Optional[Path] = None,
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
        self.path_to_baseline_model = path_to_baseline_model
        self.nav_screen = nav_screen
        self.platforms = platforms
        self.target_details = target_details
        self.target_name = target_name
        self.metric_name = metric_name
        self.session_col_name = session_col_name
        self.alpha = alpha
        self.calculate_regular_auc = calculate_regular_auc
        self.catboost_params = catboost_params
        self.formula_path = formula_path
        self.save_predictions = save_predictions

        self.use_baseline_model = bool(self.path_to_baseline_model or None)
        if np.isclose(self.alpha, 1.0) and self.use_baseline_model:
            _msg = f"When use_baseline_model = True, then alpha must be < 1.0 [alpha = {alpha:.4g}]"
            logger.error(_msg)
            raise ValueError(_msg)

        self._baseline_model_metric = (
            self._get_baseline_metric() if self.use_baseline_model else 0.0
        )
        logger.info(
            f"Use baseline model: {self.use_baseline_model} (alpha = {self.alpha:.4g})"
        )

    def _get_baseline_metric(self) -> float:
        _trainer = CatboostTrainer()
        _trainer.load(self.path_to_baseline_model)

        baseline_model_metric = get_metric(
            trainer=_trainer,
            target_config=self.target_config,
            pool_cache_info_val=self.pool_cache_info_val,
            features_pairs_generator=self.features_pairs_generator,
            nav_screen=self.nav_screen,
            platforms=self.platforms,
            formula_path=self.formula_path,
            target_details=self.target_details,
            target_name=self.target_name,
            metric_name=self.metric_name,
            session_col_name=self.session_col_name,
            calculate_regular_auc=self.calculate_regular_auc,
            model_name="baseline_model",
            save_predictions=False,
        )
        logger.info(f"Baseline model metric: {baseline_model_metric:.5f}")

        return baseline_model_metric

    def __call__(self, trial: Trial) -> float:
        like_weight = trial.suggest_float("like_weight", 0.0, 1_000.0)
        dislike_weight = trial.suggest_float("dislike_weight", 0.0, 1_000.0)
        consumption_time_weight = trial.suggest_float(
            "consumption_time_weight", 0.0, 100.0
        )
        smooth = trial.suggest_float("smooth", 0.0, 1.0)

        features_table_train = self.features_pairs_generator.generate_features_table(
            self.pool_cache_info_train.data,
            like_weight=like_weight,
            dislike_weight=dislike_weight,
            consumption_time_weight=consumption_time_weight,
            smooth=smooth,
        )
        pairs_table_train = self.features_pairs_generator.generate_pairs_table(
            features_table_train
        )

        pool_train: cb.Pool = CatBoostPoolProcessor(
            features_table_train, pairs_table_train
        ).create_pool()

        trainer = CatboostTrainer(self.catboost_params)
        trainer.train(pool_train)

        local_model_metric = get_metric(
            trainer=trainer,
            target_config=self.target_config,
            pool_cache_info_val=self.pool_cache_info_val,
            features_pairs_generator=self.features_pairs_generator,
            nav_screen=self.nav_screen,
            platforms=self.platforms,
            formula_path=self.formula_path,
            target_details=self.target_details,
            target_name=self.target_name,
            metric_name=self.metric_name,
            session_col_name=self.session_col_name,
            calculate_regular_auc=self.calculate_regular_auc,
            save_predictions=self.save_predictions,
        )

        delta = local_model_metric - self._baseline_model_metric
        logger.debug(f"Delta: {delta:.5f}")
        return self.alpha * local_model_metric + (1 - self.alpha) * (
            1 * int(self.use_baseline_model) + delta
        )
