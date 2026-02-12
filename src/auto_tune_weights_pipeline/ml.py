import typing as t

import polars as pl
import numpy as np
import catboost as cb

from loguru import logger
from pathlib import Path
from auto_tune_weights_pipeline.types_ import StrPath


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

        X: np.ndarray = self.extract_features(self.features_df["value"], n_features)
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
        keys = self.features_df["key"].to_numpy()
        unique_keys, group_ids = np.unique(keys, return_inverse=True)
        logger.info(f"Unique groups: {len(unique_keys)}")

        logger.info("Label creating ...")
        y = np.zeros(len(X), dtype=np.float32)
        for winner_idx, _ in pairs:
            y[winner_idx] += 1

        if y.max() > 0:
            y = y / y.max()

        logger.debug(f"Label stats - min: {y.min()}, max: {y.max()}, mean: {y.mean()}")
        logger.debug(
            f"Group stats - min size: {min(np.bincount(group_ids))}, max size: {max(np.bincount(group_ids))}"
        )

        logger.info("Catboost Pool creating ...")
        pool = cb.Pool(
            data=X,
            label=y,
            group_id=group_ids,
            pairs=pairs,
            feature_names=[f"f{i}" for i in range(X.shape[1])],
        )
        logger.info(
            "Data is prepared: {} samples, {} pairs, {} groups".format(
                X.shape[0], len(pairs), len(unique_keys)
            )
        )

        return pool

    @staticmethod
    def extract_features(value_series, n_features: int) -> np.ndarray:
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

    @staticmethod
    def add_catboost_scores_to_pool_cache(
        # ranker: cb.CatBoostRanker,
        model: "CatboostTrainer",
        path_to_pool_cache_val: Path,
        pool_val: cb.Pool,
        features_val: pl.DataFrame,
        score_col_name: str = "catboost_score",
        output_path: t.Optional[StrPath] = None,
        noise_coeff: float = 2,
    ) -> StrPath:
        pool_cache_val = pl.read_ndjson(str(path_to_pool_cache_val))
        logger.info(f"Loaded pool cache: {len(pool_cache_val)} rows")
        logger.debug(f"Pool cache columns: {pool_cache_val.columns}")

        logger.info("Getting predictions from ranker ...")
        predictions: np.ndarray[float] = model.get_predict(pool_val)
        logger.debug(f"Predictions shape: {predictions.shape}")
        logger.debug(f"Features_val columns: {features_val.columns}")
        logger.debug(f"Features_val row counts: {len(features_val)}")

        rid_column_in_features = None
        possible_rid_columns = ["original_rid", "rid", "session_id", "groupId"]

        for col in possible_rid_columns:
            if col in features_val.columns:
                rid_column_in_features = col
                logger.info(
                    f"Found rid column in features_val: {rid_column_in_features}"
                )
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

        pred_df = pl.DataFrame(
            {
                "rid": features_val["original_rid"],
                score_col_name: predictions
                + noise_coeff * np.random.normal(size=predictions.size),
            }
        )

        avg_scores = pred_df.group_by("rid").agg(
            pl.col(score_col_name).mean().alias(score_col_name)
        )

        pool_cache_with_scores = pool_cache_val.join(
            avg_scores,
            on="rid",
            how="left",
        )

        null_count = pool_cache_with_scores[score_col_name].null_count()
        total_rows = len(pool_cache_with_scores)

        logger.info(
            f"Result: {total_rows - null_count} rows with scores, {null_count} rows without"
        )
        logger.info(
            f"Success rate: {(total_rows - null_count) / total_rows * 100:.1f}%"
        )

        if output_path is None:
            _name = (
                path_to_pool_cache_val.name.replace(path_to_pool_cache_val.suffix, "")
                + "_with_scores"
            )
            output_path = (path_to_pool_cache_val.parent / _name).with_suffix(".jsonl")

        logger.info(f"Saving to {output_path}")
        pool_cache_with_scores.write_ndjson(output_path)

        return output_path


class CatboostTrainer:
    def __init__(
        self,
        params: dict,
        ranker_name: str = "catboost_ranker.cbm",
    ) -> None:
        self.ranker = None
        self.params = params
        self.ranker_name = ranker_name

    def train(self, pool: cb.Pool) -> None:
        logger.info("Catboost training ...")
        self.ranker = cb.CatBoostRanker(**self.params)
        self.ranker.fit(pool)

        logger.info("Ranker saving ...")
        self.ranker.save_model(self.ranker_name)

        if Path.cwd().joinpath(self.ranker_name).exists():
            logger.info(f"Ranker saved as {self.ranker_name}")
        else:
            logger.warning("Model could not be saved ...")

    def get_predict(self, pool: cb.Pool):
        return self._softmax(self.ranker.predict(pool))

    @staticmethod
    def _softmax(values) -> np.ndarray[float]:
        exp_ = np.exp(values - np.max(values))
        return exp_ / exp_.sum()
