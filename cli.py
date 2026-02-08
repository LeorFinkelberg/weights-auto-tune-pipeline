import click
import optuna
from loguru import logger

from pathlib import Path
from auto_tune_weights_pipeline.tune import Objective
from auto_tune_weights_pipeline.logging_config import setup_logging
from auto_tune_weights_pipeline.features_pairs_generator import FeaturesPairsGenerator
from auto_tune_weights_pipeline.constants import LossFunctions


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
@click.option("--n-trials", type=click.INT, default=3)
@click.option("--timeout", type=click.FLOAT, default=120)
@click.option("--direction", type=click.STRING, default="maximize")
@click.option("--study-name", type=click.STRING, default="tune_target_events_weights")
@click.option("--load-if-exists", type=click.BOOL, default=True)
@click.option("--gc-after-trial", type=click.BOOL, default=True)
@click.option("--show-progress-bar", type=click.BOOL, default=True)
def main(
    path_to_pool_cache_train,
    path_to_pool_cache_val,
    path_to_feature_names,
    n_trials,
    timeout,
    direction,
    study_name,
    load_if_exists,
    gc_after_trial,
    show_progress_bar,
) -> None:
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",
        load_if_exists=load_if_exists,
    )

    study.optimize(
        Objective(
            path_to_pool_cache_train=path_to_pool_cache_train,
            path_to_pool_cache_val=path_to_pool_cache_val,
            feature_pairs_generator=FeaturesPairsGenerator(
                path_to_feature_names=path_to_feature_names,
            ),
            catboost_params={
                "iterations": 10,
                "l2_leaf_reg": 3,
                "learning_rate": 0.05,
                "border_count": 32,
                "subsample": 0.7,
                "loss_function": LossFunctions.PAIR_LOGIT_PAIRWISE,
            },
        ),
        n_trials=n_trials,
        timeout=timeout,
        gc_after_trial=gc_after_trial,
        show_progress_bar=show_progress_bar,
    )

    logger.info(f"Best value: {study.best_trial.value}")
    logger.info(f"Best trial number: {study.best_trial.number}")
    logger.info(f"Best trial params: {study.best_trial.params}")


if __name__ == "__main__":
    main()
