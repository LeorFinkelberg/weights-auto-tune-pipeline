import typing as t

import polars as pl
import numpy as np

from typing_extensions import override
from loguru import logger
from tqdm import tqdm

from auto_tune_weights_pipeline.columns import Columns
from auto_tune_weights_pipeline.logging_ import setup_logging
from auto_tune_weights_pipeline.metrics.base import Metric
from auto_tune_weights_pipeline.target_config import (
    TargetConfig,
    DEFAULT_TARGETS_CONFIG,
)

setup_logging()


class GAUC(Metric):
    """Group AUC metric calculator."""

    @override
    def calculate_metric(
        self,
        target_configs: t.Union[list[TargetConfig], list[str], dict[str, TargetConfig]],
        session_col_name: str = Columns.RID_COL_NAME,
        score_col_name: str = Columns.SCORE_COL_NAME,
        platform: str = "vk_video_android",
        nav_screen: str = "video_for_you",
        formula_path: str = "fstorage:vk_video_266_1769078359_f",
        nav_screen_col_name: str = Columns.NAV_SCREEN_COL_NAME,
        platform_col_name: str = Columns.PLATFORM_COL_NAME,
        formula_path_col_name: str = Columns.FORMULA_PATH_COL_NAME,
    ) -> dict[str, dict[str, t.Any]]:
        """Calculates metric."""

        _mask = (
            (pl.col(nav_screen_col_name) == nav_screen)
            & (pl.col(platform_col_name) == platform)
            & (pl.col(formula_path_col_name) == formula_path)
        )
        pool_cache: pl.DataFrame = pl.read_ndjson(self.path_to_pool_cache).filter(_mask)

        return self._calculate_metric_for_all_targets(
            pool_cache,
            target_configs=target_configs,
            session_col_name=session_col_name,
            score_col_name=score_col_name,
        )

    @staticmethod
    def _create_labels(
        pool_cache: pl.DataFrame,
        target_configs: t.Union[
            list[TargetConfig],
            list[str],
            dict[str, TargetConfig],
        ],
    ) -> pl.DataFrame:
        """Creates labels."""

        if isinstance(target_configs, dict):
            configs = list(target_configs.values())
        elif isinstance(target_configs, list):
            if all(isinstance(config, str) for config in target_configs):
                configs = [DEFAULT_TARGETS_CONFIG[name] for name in target_configs]
            else:
                configs = target_configs
        else:
            raise TypeError("target_configs must be list or dict ...")

        logger.info("Label creating ...")
        label_exprs = []
        for config in configs:
            label_expr = config.create_label_expr(pool_cache).alias(
                f"label_{config.name}"
            )
            label_exprs.append(label_expr)

        return pool_cache.with_columns(label_exprs)

    def _calculate_metric_for_all_targets(
        self,
        pool_cache: pl.DataFrame,
        target_configs: t.Union[list[TargetConfig], list[str], dict[str, TargetConfig]],
        session_col_name: str = Columns.RID_COL_NAME,
        score_col_name: str = Columns.SCORE_COL_NAME,
    ) -> dict[str, dict[str, t.Any]]:
        """Calculates metric for all targets."""

        pool_cache_with_labels = self._create_labels(pool_cache, target_configs)

        if isinstance(target_configs, dict):
            target_names = list(target_configs.keys())
        elif isinstance(target_configs, list):
            if all(isinstance(target_config, str) for target_config in target_configs):
                target_names = target_configs
            else:
                target_names = [target_config.name for target_config in target_configs]

        results = {}
        for target_name in target_names:
            result = self._calculate_metric_for_target_chunked(
                pool_cache_with_labels,
                target_name,
                session_col_name,
                score_col_name,
            )
            results[target_name] = result

        return results

    def _calculate_metric_for_target_chunked(
        self,
        pool_cache: pl.DataFrame,
        target_name: str,
        session_col_name: str,
        score_col_name: str,
        chunk_size: int = 100_000,
    ) -> dict[str, t.Any]:
        """Calculates metric for target by chunks."""

        label_col_name = f"label_{target_name}"
        logger.info(
            f"Calculating CHUNKED {self.get_metric_name()} for target: {target_name}"
        )

        all_auc_values = []
        all_group_sizes = []
        all_group_details = []

        unique_groups = pool_cache[session_col_name].unique().to_list()
        logger.info(f"Total unique groups: {len(unique_groups)}")

        for i in tqdm(
            range(0, len(unique_groups), chunk_size), desc="Groups processing ..."
        ):
            chunk_groups = unique_groups[i : i + chunk_size]

            chunk_pool_cache = pool_cache.filter(
                pl.col(session_col_name).is_in(chunk_groups)
            )

            chunk_stats = (
                chunk_pool_cache.group_by(session_col_name)
                .agg(
                    [
                        pl.col(label_col_name).sum().alias("n_pos"),
                        pl.col(label_col_name).count().alias("n_total"),
                        pl.col(label_col_name).min().alias("min_label"),
                        pl.col(label_col_name).max().alias("max_label"),
                    ]
                )
                .filter(
                    (pl.col("min_label") == 0)
                    & (pl.col("max_label") == 1)
                    & (pl.col("n_total") >= 2)
                    & (pl.col("n_pos") > 0)
                    & (pl.col("n_pos") < pl.col("n_total"))
                )
            )

            if len(chunk_stats) == 0:
                continue

            for row in chunk_stats.iter_rows(named=True):
                session_id = row[session_col_name]

                group_data = chunk_pool_cache.filter(
                    pl.col(session_col_name) == session_id
                )

                y_true = group_data[label_col_name].to_numpy()
                y_score = group_data[score_col_name].to_numpy()

                order = np.argsort(y_score)[::-1]
                y_true_sorted = y_true[order]

                n_pos = np.sum(y_true_sorted)
                n_neg = len(y_true_sorted) - n_pos

                pos_ranks = np.where(y_true_sorted == 1)[0] + 1
                auc = (np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

                all_auc_values.append(auc)
                all_group_sizes.append(len(group_data))

                all_group_details.append(
                    {
                        "session_id": session_id,
                        "auc": float(auc),
                        "size": len(group_data),
                        "positives": int(n_pos),
                        "negatives": int(n_neg),
                        "positive_rate": float(n_pos / len(group_data)),
                    }
                )

            logger.info(
                f"Processed {min(i + chunk_size, len(unique_groups))} / {len(unique_groups)} groups"
            )

        if not all_auc_values:
            return self._empty_result(pool_cache, target_name, label_col_name)

        auc_array = np.array(all_auc_values)
        weights_array = np.array(all_group_sizes)

        result = {
            "target": target_name,
            "GAUCSimple": float(np.mean(auc_array)),
            "GAUCWeighted": float(np.average(auc_array, weights=weights_array)),
            "n_groups": len(auc_array),
            "std": float(np.std(auc_array)),
            "min": float(np.min(auc_array)),
            "max": float(np.max(auc_array)),
            "median": float(np.median(auc_array)),
            "total_samples": len(pool_cache),
            "total_positives": pool_cache[label_col_name].sum(),
            "positive_rate": float(pool_cache[label_col_name].mean()),
            "group_details": all_group_details,
        }

        self._log_results(result)
        return result

    @staticmethod
    def _empty_result(
        pool_cache: pl.DataFrame, target_name: str, label_col_name: str
    ) -> dict[str, t.Any]:
        """Returns empty result."""

        return {
            "target": target_name,
            "GAUCSimple": 0.0,
            "GAUCWeighted": 0.0,
            "n_groups": 0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "total_samples": len(pool_cache),
            "total_positives": pool_cache[label_col_name].sum(),
            "positive_rate": float(pool_cache[label_col_name].mean()),
            "group_details": [],
        }

    @staticmethod
    def _log_results(result: dict[str, t.Any]) -> None:
        """Logs result."""

        logger.info(f"Groups with AUC: {result['n_groups']}")
        logger.info(
            f"Positive samples: {result['total_positives']} ({result['positive_rate']:.1%})"
        )
        logger.info(f"GAUC (weighted): {result['GAUCWeighted']:.4f}")
        logger.info(f"GAUC (simple): {result['GAUCSimple']:.4f}")
        logger.info(f"Std AUC: {result['std']:.4f}")
