import polars as pl
import numpy as np
import catboost as cb

from loguru import logger
from auto_tune_weights_pipeline.constants import (
    RANDOM_SEED,
    LossFunctions,
    CatboostTaskTypes,
)
from pathlib import Path


class CatBoostPoolProcessor:
    def __init__(
        self,
        features_df: pl.DataFrame,
        pairs_df: pl.DataFrame,
    ) -> None:
        self.features_df = features_df
        self.pairs_df = pairs_df

    def create_pool(self) -> cb.Pool:
        logger.info("Data preparing for CatBoost ...")

        logger.info("Feature extracting ...")
        sample = self.features_df["value"][0]
        n_features = len(sample.split("\t")) - 2

        X: np.ndarray = self._extract_features(self.features_df["value"], n_features)
        logger.info(f"Feature matrix: {X.shape}")

        logger.info("Pairs extracting ...")
        pairs = []

        for row in self.pairs_df.iter_rows(named=True):
            try:
                winner_idx = int(row["key"])

                value = row["value"]
                parts = value.split("\t")
                if len(parts) < 1:
                    continue

                looser_idx = int(parts[0])
                if 0 <= winner_idx < len(X) and 0 <= looser_idx < len(X):
                    pairs.append((winner_idx, looser_idx))

            except (ValueError, TypeError):
                continue

        logger.info(f"Extracted pairs: {len(pairs)}")

        logger.info("Group creating ...")
        group_id = np.zeros(len(X), dtype=np.int32)

        logger.info("Label creating ...")
        y = np.zeros(len(X), dtype=np.float32)
        for winner_idx, _ in pairs:
            y[winner_idx] += 1

        if y.max() > 0:
            y = y / y.max()

        logger.info("Catboost Pool creating ...")
        pool = cb.Pool(
            data=X,
            label=y,
            group_id=group_id,
            pairs=pairs,
            feature_names=[f"f{i}" for i in range(X.shape[1])],
        )
        logger.info(
            "Data is prepared: {} samples, {} pairs".format(X.shape[0], len(pairs))
        )

        return pool

    @staticmethod
    def _extract_features(value_series, n_features: int) -> np.ndarray:
        features_list = []

        for value in value_series:
            parts = value.split("\t")
            if len(parts) < 3:
                features_list.append([0.0] * n_features)
                continue

            feature_strs = parts[2:]
            features = []

            for feat_str in feature_strs[:n_features]:
                try:
                    features.append(float(feat_str))
                except (ValueError, TypeError):
                    features.append(float("nan"))

            if len(features) < n_features:
                features.extend([0.0] * (n_features - len(features)))

            features_list.append(features)

        return np.array(features_list, dtype=np.float32)


class CatboostTrainer:
    def __init__(
        self,
        iterations: int = 300,
        learning_rate: float = 0.05,
        depth: int = 6,
        loss_function: str = LossFunctions.PAIR_LOGIT,
        verbose=100,
        random_seed: int = RANDOM_SEED,
        task_type: CatboostTaskTypes = CatboostTaskTypes.CPU,
        ranker_name: str = "catboost_ranker.cbm",
    ) -> None:
        self.ranker = None
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.loss_function = loss_function
        self.verbose = verbose
        self.random_seed = random_seed
        self.task_type = task_type
        self.ranker_name = ranker_name

    def train(self, pool: cb.Pool) -> None:
        logger.info("Catboost training ...")
        self.ranker = cb.CatBoostRanker(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function=self.loss_function,
            verbose=self.verbose,
            random_seed=self.random_seed,
            task_type=self.task_type,
        )
        self.ranker.fit(pool)

        logger.info("Ranker saving ...")
        self.ranker.save_model(self.ranker_name)

        if Path.cwd().joinpath(self.ranker_name).exists():
            logger.info(f"Ranker saved as {self.ranker_name}")
        else:
            logger.warning("Model could not be saved ...")

    def predict(self, pool: cb.Pool):
        return self._softmax(self.ranker.predict(pool))

    @staticmethod
    def _softmax(values) -> float:
        exp_ = np.exp(values - np.max(values))
        return exp_ / exp_.sum()
