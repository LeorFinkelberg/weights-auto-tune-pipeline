import typing as t

import polars as pl
import click
import optuna

from loguru import logger
from pathlib import Path

from auto_tune_weights_pipeline.event_names import EventNames
from auto_tune_weights_pipeline.tune import Objective
from auto_tune_weights_pipeline.logging_config import setup_logging
from auto_tune_weights_pipeline.features_pairs_generator import FeaturesPairsGenerator
from auto_tune_weights_pipeline.constants import (
    LossFunctions,
    Platforms,
    NavScreens,
    SummaryLogFields,
)
from auto_tune_weights_pipeline.ml import PoolCacheInfo, CatboostTrainer
from auto_tune_weights_pipeline.target_config import TargetConfig, TargetNames
from auto_tune_weights_pipeline.metrics.utils import get_metric
from auto_tune_weights_pipeline.columns import Columns

setup_logging()


@click.command(help="Tune weights for target events")
@click.option(
    "--path-to-pool-cache-train",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
)
@click.option(
    "--path-to-pool-cache-val",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
)
@click.option(
    "--path-to-feature-names",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    default="./feature_names.txt",
)
@click.option(
    "--path-to-pretrained-model",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    default=None,
)
@click.option(
    "--path-to-baseline-model",
    type=click.Path(file_okay=True, path_type=Path),
    default=None,
)
@click.option("--nav-screen", type=click.STRING, default=NavScreens.VIDEO_FOR_YOU)
@click.option(
    "--platforms",
    type=click.STRING,
    default=(Platforms.ANDROID, Platforms.VK_VIDEO_ANDROID),
    multiple=True,
)
@click.option(
    "--formula-path", type=click.STRING, default="fstorage:vk_video_266_1769078359_f"
)
@click.option(
    "--target-details", type=click.STRING, default=SummaryLogFields.TARGET_DETAILS
)
@click.option(
    "--target-name", type=click.STRING, default=TargetNames.WATCH_COVERAGE_30S
)
@click.option("--metric-name", default=SummaryLogFields.GAUC_WEIGHTED)
@click.option(
    "--session-col-name",
    type=click.Choice([Columns.RID_COL_NAME.value, Columns.USER_ID_COL_NAME.value]),
    default=Columns.RID_COL_NAME,
)
@click.option("--alpha", type=click.FLOAT, default=1.0)
@click.option("--iterations", type=click.INT, default=150)
@click.option("--depth", type=click.INT, default=6)
@click.option("--l2-leaf-reg", type=click.FLOAT, default=3.0)
@click.option("--learning-rate", type=click.FLOAT, default=0.05)
@click.option("--subsample", type=click.FLOAT, default=0.7)
@click.option(
    "--loss-function", type=click.STRING, default=LossFunctions.PAIR_LOGIT_PAIRWISE
)
@click.option("--n-trials", type=click.INT, default=3)
@click.option("--timeout", type=click.FLOAT, default=120)
@click.option("--direction", type=click.STRING, default="maximize")
@click.option("--study-name", type=click.STRING, default="tune_target_events_weights")
@click.option("--load-if-exists/--no-load-if-exists", default=True)
@click.option("--gc-after-trial/--no-gc-after-trial", default=True)
@click.option("--show-progress-bar/--no-show-progress-bar", default=True)
@click.option("--save-predictions/--no-save-predictions", default=False)
@click.option("--calculate-regular-auc/--no-calculate-regular-auc", default=True)
def main(
    path_to_pool_cache_train,
    path_to_pool_cache_val,
    path_to_feature_names,
    path_to_pretrained_model,
    path_to_baseline_model,
    nav_screen,
    platforms,
    formula_path,
    target_details,
    target_name,
    metric_name,
    session_col_name,
    alpha,
    iterations,
    depth,
    l2_leaf_reg,
    learning_rate,
    subsample,
    loss_function,
    n_trials,
    timeout,
    direction,
    study_name,
    load_if_exists,
    gc_after_trial,
    show_progress_bar,
    save_predictions,
    calculate_regular_auc,
) -> None:
    target_config: t.Final[dict] = {
        TargetNames.WATCH_COVERAGE_30S: TargetConfig(
            target_name=TargetNames.WATCH_COVERAGE_30S,
            event_name=EventNames.WATCH_COVERAGE_RECORD,
            view_threshold_sec=30.0,
        )
    }

    pool_cache_info_val = PoolCacheInfo(
        data=pl.read_ndjson(path_to_pool_cache_val),
        path_to_data=path_to_pool_cache_val,
    )
    features_pairs_generator = FeaturesPairsGenerator(
        path_to_feature_names=path_to_feature_names
    )

    if path_to_pretrained_model is None:
        msg = (
            f"=== TARGET EVENT WEIGHTING MODE (GroupBy: {str(session_col_name)!r}) ==="
        )
        logger.info(msg)

        pool_cache_info_train = PoolCacheInfo(
            data=pl.read_ndjson(path_to_pool_cache_train),
            path_to_data=path_to_pool_cache_train,
        )

        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=f"sqlite:///{study_name}.db",
            load_if_exists=load_if_exists,
        )

        study.optimize(
            Objective(
                target_config,
                pool_cache_info_train=pool_cache_info_train,
                pool_cache_info_val=pool_cache_info_val,
                path_to_baseline_model=path_to_baseline_model,
                features_pairs_generator=features_pairs_generator,
                catboost_params={
                    "iterations": iterations,
                    "depth": depth,
                    "l2_leaf_reg": l2_leaf_reg,
                    "learning_rate": learning_rate,
                    "subsample": subsample,
                    "loss_function": loss_function,
                },
                nav_screen=nav_screen,
                platforms=platforms,
                formula_path=formula_path,
                target_details=target_details,
                target_name=target_name,
                metric_name=metric_name,
                session_col_name=session_col_name,
                alpha=alpha,
                calculate_regular_auc=calculate_regular_auc,
                save_predictions=save_predictions,
            ),
            n_trials=n_trials,
            timeout=timeout,
            gc_after_trial=gc_after_trial,
            show_progress_bar=show_progress_bar,
        )

        logger.info(f"Best value: {study.best_trial.value}")
        logger.info(f"Best trial number: {study.best_trial.number}")
        logger.info(f"Best trial params: {study.best_trial.params}")
    else:
        logger.info("=== MODE FOR OBTAINING FORECASTS FROM PRE-TRAINED MODEL ===")

        trainer = CatboostTrainer()
        trainer.load(path_to_pretrained_model)

        _model_name = path_to_pretrained_model.name.replace(
            path_to_pretrained_model.suffix, ""
        )
        _ = get_metric(
            trainer=trainer,
            target_config=target_config,
            pool_cache_info_val=pool_cache_info_val,
            features_pairs_generator=features_pairs_generator,
            nav_screen=nav_screen,
            platforms=platforms,
            formula_path=formula_path,
            target_details=target_details,
            target_name=target_name,
            metric_name=metric_name,
            session_col_name=session_col_name,
            calculate_regular_auc=calculate_regular_auc,
            model_name=_model_name,
            save_predictions=save_predictions,
        )


if __name__ == "__main__":
    main()
