import typing as t
import polars as pl
import numpy as np
import catboost as cb

from loguru import logger
from pathlib import Path
from dataclasses import dataclass

from auto_tune_weights_pipeline.columns import Columns
from auto_tune_weights_pipeline.constants import BIG_NEGATIVE_DEFAULT_VALUE
from auto_tune_weights_pipeline.types_ import StrPath
from auto_tune_weights_pipeline.features_pairs_generator import FeaturesPairsGenerator


@dataclass(frozen=True)
class PoolCacheInfo:
    data: pl.DataFrame
    path_to_data: StrPath


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
        pairs_weight = []

        for row in self.pairs_df.iter_rows(named=True):
            try:
                winner_idx = int(row["key"])
                value = row["value"]
                parts = value.split("\t")
                if len(parts) < 2:
                    continue

                looser_idx = int(parts[0])
                weight = float(parts[1])

                if 0 <= winner_idx < len(X) and 0 <= looser_idx < len(X):
                    pairs.append((winner_idx, looser_idx))
                    pairs_weight.append(weight)
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
            pairs_weight=pairs_weight,
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
        trainer: "CatboostTrainer",
        pool_cache_info_val: PoolCacheInfo,
        features_pairs_generator: FeaturesPairsGenerator,
        score_col_name: str = Columns.CATBOOST_SCORE_COL_NAME,
        output_path: t.Optional[StrPath] = None,
    ) -> Path:
        pool_cache_val = pool_cache_info_val.data
        logger.info(f"Loaded pool cache: {len(pool_cache_val)} rows")

        raw_features_list = pool_cache_val[Columns.FEATURES_COL_NAME].to_list()
        logger.info(f"Extracted {len(raw_features_list)} raw feature records")

        features_dicts = [
            features_pairs_generator.extract_features_dict(feat)
            for feat in raw_features_list
        ]
        logger.info(f"Converted to {len(features_dicts)} feature dicts")

        feature_order = features_pairs_generator.features
        X_list = []
        for feat_dict in features_dicts:
            row = [
                feat_dict.get(fid, BIG_NEGATIVE_DEFAULT_VALUE) for fid in feature_order
            ]
            X_list.append(row)

        X = np.array(X_list, dtype=np.float32)
        logger.info(f"Feature matrix shape: {X.shape}")

        predictions = trainer.predict(X)
        logger.info(f"Got predictions: {len(predictions)}")

        pool_cache_with_scores = pool_cache_val.with_columns(
            pl.Series(score_col_name, predictions)
        )

        if output_path is None:
            path_to_pool_cache_val = pool_cache_info_val.path_to_data
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
        params: t.Optional[dict] = None,
        ranker_name: str = "catboost_ranker.cbm",
    ) -> None:
        self.params = params or {}
        self.ranker = None
        self.params = params
        self.ranker_name = ranker_name

    def train(self, pool: cb.Pool) -> None:
        logger.info("Catboost training ...")
        self.ranker = cb.CatBoostRanker(**self.params)
        self.ranker.fit(pool)

        logger.info("Ranker saving ...")
        _path_to_data = Path.cwd().joinpath("data")
        if not _path_to_data.exists():
            _path_to_data.mkdir(exist_ok=True)

        self.ranker.save_model(str(_path_to_data / self.ranker_name))

        if (_path_to_data / self.ranker_name).exists():
            logger.info(f"Ranker saved as {self.ranker_name}")
        else:
            logger.warning("Model could not be saved ...")

    def predict(self, X: np.ndarray, noise: float = 0.0) -> np.ndarray:
        if self.ranker is None:
            raise ValueError("Model not trained yet!")

        predictions = self.ranker.predict(X)
        predictions = np.nan_to_num(
            predictions + noise * np.random.normal(size=predictions.size),
            nan=BIG_NEGATIVE_DEFAULT_VALUE,
        )

        logger.debug(
            f"Raw predictions - min: {predictions.min():.4f}, "
            f"max: {predictions.max():.4f}, "
            f"std: {predictions.std():.4f}"
        )

        return predictions

    def load(self, path_to_model: StrPath) -> None:
        self.ranker = cb.CatBoostRanker().load_model(str(path_to_model))
