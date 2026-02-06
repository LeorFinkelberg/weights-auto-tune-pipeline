import typing as t
import polars as pl
import numpy as np
from pathlib import Path
from loguru import logger


class MatrixBuilder:
    def __init__(self, path_to_pool_cache_with_features: t.Union[str, Path]):
        self.data_path = Path(path_to_pool_cache_with_features)
        self.df: t.Optional[pl.DataFrame] = None
        self._load_data()

    def _load_data(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"File not found: {self.data_path}")

        logger.info(f"Loading {self.data_path}")
        self.df = pl.read_ndjson(str(self.data_path))
        logger.info(f"Loaded {len(self.df)} rows")

    def get_Xy_matrix(
        self,
        target_label: str = "actionPlay",
        view_time_threshold: t.Optional[int] = None,
        num_features: int = 256,
    ) -> t.Tuple[pl.DataFrame, pl.Series]:
        if self.df is None:
            raise ValueError("Data not loaded")

        logger.info(f"Building matrix for: {target_label}")

        X_df = self._build_feature_matrix(num_features)
        y_series = self._create_labels_series(target_label, view_time_threshold)

        logger.info(f"X shape: {X_df.shape}, y positive: {y_series.mean():.2%}")

        return X_df, y_series

    def _create_labels_series(
        self, target_label: str, view_time_threshold: t.Optional[int]
    ) -> pl.Series:
        has_event = (
            self.df["events"]
            .list.eval(pl.element().str.contains(target_label))
            .list.any()
        )

        if view_time_threshold is not None:
            has_time = pl.col("viewTimeSec") >= view_time_threshold
            label = (has_event & has_time).cast(pl.Int8)
        else:
            label = has_event.cast(pl.Int8)

        pos_count = label.sum()
        total = len(self.df)
        logger.info(f"Labels: {pos_count}/{total} positive ({pos_count / total:.1%})")

        return label

    def _build_feature_matrix(self, num_features: int) -> pl.DataFrame:
        logger.info(f"Building feature matrix with {num_features} features")

        num_rows = len(self.df)
        feature_matrix = np.zeros((num_rows, num_features), dtype=np.float32)

        features_column = self.df["features"].to_list()

        for row_idx, feature_pairs in enumerate(features_column):
            if feature_pairs is None:
                continue

            for pair in feature_pairs:
                if (
                    isinstance(pair, list)
                    and len(pair) >= 2
                    and isinstance(pair[0], (int, float))
                ):

                    try:
                        feat_idx = int(pair[0])
                        feat_value = float(pair[1])

                        if 0 <= feat_idx < num_features:
                            feature_matrix[row_idx, feat_idx] = feat_value
                        else:
                            logger.debug(
                                f"Feature index {feat_idx} out of bounds [0, {num_features})"
                            )

                    except (ValueError, TypeError) as e:
                        logger.debug(f"Error parsing feature pair: {pair}, error: {e}")
                        continue

        columns = {f"feat_{i}": feature_matrix[:, i] for i in range(num_features)}
        X_df = pl.DataFrame(columns)

        non_zero = np.count_nonzero(feature_matrix)
        total_cells = num_rows * num_features
        logger.info(
            f"Non-zero values: {non_zero}/{total_cells} ({non_zero / total_cells:.2%})"
        )

        used_features = np.any(feature_matrix != 0, axis=0)
        used_count = np.sum(used_features)
        logger.info(f"Used features: {used_count}/{num_features}")

        if used_count < num_features:
            unused = np.where(~used_features)[0]
            logger.info(f"Unused feature indices (first 10): {unused[:10].tolist()}")

        return X_df

    def analyze_features(self, num_features: int = 256) -> t.Dict[str, t.Any]:
        if self.df is None:
            return {}

        features_column = self.df["features"].to_list()

        feature_stats = {}
        total_pairs = 0

        for feature_pairs in features_column:
            if feature_pairs is None:
                continue

            total_pairs += len(feature_pairs)

            for pair in feature_pairs:
                if (
                    isinstance(pair, list)
                    and len(pair) >= 2
                    and isinstance(pair[0], (int, float))
                ):

                    try:
                        feat_idx = int(pair[0])
                        feat_value = float(pair[1])

                        if feat_idx not in feature_stats:
                            feature_stats[feat_idx] = {"count": 0, "values": []}

                        feature_stats[feat_idx]["count"] += 1
                        feature_stats[feat_idx]["values"].append(feat_value)

                    except (ValueError, TypeError):
                        continue

        result = {
            "total_rows": len(features_column),
            "total_feature_pairs": total_pairs,
            "avg_pairs_per_row": (
                total_pairs / len(features_column) if features_column else 0
            ),
            "feature_distribution": {},
            "max_feature_index": max(feature_stats.keys()) if feature_stats else 0,
        }

        for idx, stats in feature_stats.items():
            if idx < num_features:
                values = stats["values"]
                result["feature_distribution"][idx] = {
                    "count": stats["count"],
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }

        return result


class AutoMatrixBuilder(MatrixBuilder):
    def get_Xy_matrix(
        self,
        target_label: str = "actionPlay",
        view_time_threshold: t.Optional[int] = None,
        max_features: int = 500,
        min_frequency: int = 10,
    ) -> t.Tuple[pl.DataFrame, pl.Series]:
        if self.df is None:
            raise ValueError("Data not loaded")

        logger.info(f"Building matrix for: {target_label}")

        feature_indices = self._get_used_feature_indices(min_frequency)
        y_series = self._create_labels_series(target_label, view_time_threshold)

        if len(feature_indices) > max_features:
            logger.info(
                f"Limiting features from {len(feature_indices)} to {max_features}"
            )
            feature_indices = feature_indices[:max_features]

        X_df = self._build_selected_feature_matrix(feature_indices)

        logger.info(f"X shape: {X_df.shape}, using {len(feature_indices)} features")
        logger.info(
            f"Feature indices: {feature_indices[:20]}..."
            if len(feature_indices) > 20
            else f"Feature indices: {feature_indices}"
        )

        return X_df, y_series

    def _get_used_feature_indices(self, min_frequency: int) -> t.List[int]:
        features_column = self.df["features"].to_list()

        feature_counts = {}

        for feature_pairs in features_column:
            if feature_pairs is None:
                continue

            for pair in feature_pairs:
                if (
                    isinstance(pair, list)
                    and len(pair) >= 2
                    and isinstance(pair[0], (int, float))
                ):

                    try:
                        feat_idx = int(pair[0])
                        feature_counts[feat_idx] = feature_counts.get(feat_idx, 0) + 1
                    except (ValueError, TypeError):
                        continue

        used_indices = [
            idx for idx, count in feature_counts.items() if count >= min_frequency
        ]

        used_indices.sort()

        logger.info(
            f"Found {len(used_indices)} features with frequency >= {min_frequency}"
        )
        logger.info(
            f"Feature frequency range: {min(feature_counts.values()) if feature_counts else 0} - {max(feature_counts.values()) if feature_counts else 0}"
        )

        return used_indices

    def _build_selected_feature_matrix(
        self, feature_indices: list[int]
    ) -> pl.DataFrame:
        num_rows = len(self.df)
        num_features = len(feature_indices)

        idx_to_position = {idx: pos for pos, idx in enumerate(feature_indices)}

        feature_matrix = np.zeros((num_rows, num_features), dtype=np.float32)
        features_column = self.df["features"].to_list()

        for row_idx, feature_pairs in enumerate(features_column):
            if feature_pairs is None:
                continue

            for pair in feature_pairs:
                if (
                    isinstance(pair, list)
                    and len(pair) >= 2
                    and isinstance(pair[0], (int, float))
                ):

                    try:
                        feat_idx = int(pair[0])
                        feat_value = float(pair[1])

                        if feat_idx in idx_to_position:
                            pos = idx_to_position[feat_idx]
                            feature_matrix[row_idx, pos] = feat_value

                    except (ValueError, TypeError):
                        continue

        columns = {
            f"feat_{feature_indices[i]}": feature_matrix[:, i]
            for i in range(num_features)
        }

        return pl.DataFrame(columns)
