import typing as t

import polars as pl
import numpy as np

from typing_extensions import override
from sklearn.metrics import roc_auc_score
from loguru import logger

from auto_tune_weights_pipeline.columns import Columns
from auto_tune_weights_pipeline.logging_ import setup_logging
from auto_tune_weights_pipeline.metrics.base import Metric
from auto_tune_weights_pipeline.target_config import (
    TargetConfig,
    DEFAULT_TARGETS_CONFIG,
)

setup_logging()


class Gauc(Metric):
    """..."""

    @override
    def calculate_metric(
        self,
        target_configs: t.Union[list[TargetConfig], list[str], dict[str, TargetConfig]],
        session_col_name: str = Columns.RID_COL_NAME,
        score_col_name: str = Columns.SCORE_COL_NAME,
        nav_screen: str = "feed",
        platform: str = "vk_video_android",
        formula_path: str = "fstorage:vk_video_266_1769078359_f",
        nav_screen_col_name: str = Columns.NAV_SCREEN_COL_NAME,
        platform_col_name: str = Columns.PLATFORM_COL_NAME,
        formula_path_col_name: str = Columns.FORMULA_PATH_COL_NAME,
    ) -> dict[str, dict[str, t.Any]]:
        pool_cache = pl.read_ndjson(self.path_to_pool_cache)
        pool_cache = pool_cache.filter(
            (pl.col(Columns.NAV_SCREEN_COL_NAME) == nav_screen)
            & (pl.col(Columns.PLATFORM_COL_NAME) == platform)
        )

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

    def _calculate_metric_for_target(
        self,
        pool_cache: pl.DataFrame,
        target_name: str,
        session_col_name: str,
        score_col_name: str,
    ) -> dict[str, t.Any]:
        """Computes metric."""

        label_col_name = f"label_{target_name}"
        if label_col_name not in pool_cache.columns:
            raise ValueError(f"Label {label_col_name!r} not found ...")

        logger.info(
            f"Run calculating {self.get_metric_name()} for target event: {target_name}"
        )
        group_stats = pool_cache.group_by(session_col_name).agg(
            [
                pl.col(label_col_name).sum().alias("pos_count"),
                pl.col(label_col_name).count().alias("total_count"),
                pl.col(label_col_name).min().alias("min_label"),
                pl.col(label_col_name).max().alias("max_label"),
            ]
        )

        valid_groups = group_stats.filter(
            (pl.col("min_label") == 0) & (pl.col("max_label") == 1)
        )[session_col_name]

        pool_cache_valid = pool_cache.filter(
            pl.col(session_col_name).is_in(valid_groups)
        )
        logger.info(f"Groups count: {pool_cache[session_col_name].n_unique()}")
        logger.info(f"Valid groups count: {len(valid_groups)}")

        auc_results = []
        group_sizes = []
        group_details = []

        for session_id in valid_groups.to_list():
            group_pool_cache = pool_cache_valid.filter(
                pl.col(session_col_name) == session_id
            )

            if len(group_pool_cache) < 2:
                continue

            y_true = group_pool_cache[label_col_name].to_numpy()
            y_score = group_pool_cache[score_col_name].to_numpy()

            if len(np.unique(y_true)) < 2:
                continue

            try:
                auc = roc_auc_score(y_true, y_score)
                auc_results.append(auc)
                group_sizes.append(len(group_pool_cache))

                group_details.append(
                    {
                        "session_id": session_id,
                        "auc": auc,
                        "size": len(group_pool_cache),
                        "positives": int(y_true.sum()),
                        "negatives": len(y_true) - int(y_true.sum()),
                        "positive_rate": float(y_true.mean()),
                    }
                )
            except Exception as err:
                logger.error(f"Error in group {session_id}: {err}")
                continue

        if not auc_results:
            logger.debug(pool_cache.count())
            logger.warning("There are no valid groups for calculating AUC")
            return {
                "target": target_name,
                "gauc_simple": 0.0,
                "gauc_weighted": 0.0,
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

        auc_array = np.array(auc_results)

        result = {
            "target": target_name,
            "gauc_simple": float(np.mean(auc_array)),
            "gauc_weighted": float(np.average(auc_array, weights=group_sizes)),
            "n_groups": len(auc_results),
            "std": float(np.std(auc_array)),
            "min": float(np.min(auc_array)),
            "max": float(np.max(auc_array)),
            "median": float(np.median(auc_array)),
            "total_samples": len(pool_cache),
            "total_positives": int(pool_cache[label_col_name].sum()),
            "positive_rate": float(pool_cache[label_col_name].mean()),
            "group_details": group_details,
        }

        logger.info(f"Groups with AUC: {result['n_groups']}")
        logger.info(
            f"Positive samples: {result['total_positives']} ({result['positive_rate']:.1%})"
        )
        logger.info(f"GAUC (weighted): {result['gauc_weighted']:.4f}")
        logger.info(f"GAUC: {result['gauc_simple']:.4f}")
        logger.info(f"Std AUC: {result['std']:.4f}")

        return result

    def _calculate_metric_for_all_targets(
        self,
        pool_cache: pl.DataFrame,
        target_configs: t.Union[list[TargetConfig], list[str], dict[str, TargetConfig]],
        session_col_name: str = Columns.RID_COL_NAME,
        score_col_name: str = Columns.SCORE_COL_NAME,
    ) -> dict[str, dict[str, t.Any]]:
        pool_cache_with_labels = self._create_labels(pool_cache, target_configs)

        if isinstance(target_configs, dict):
            target_names = list(target_configs.keys())
        elif isinstance(target_configs, list):
            if all(isinstance(tc, str) for tc in target_configs):
                target_names = target_configs
            else:
                target_names = [tc.name for tc in target_configs]

        results = {}
        for target_name in target_names:
            result = self._calculate_metric_for_target(
                pool_cache_with_labels,
                target_name,
                session_col_name,
                score_col_name,
            )
            results[target_name] = result

        return results
