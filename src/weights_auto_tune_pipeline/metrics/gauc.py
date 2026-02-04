import typing as t

import polars as pl
import numpy as np

from base import Metric
from typing_extensions import override
from sklearn.metrics import roc_auc_score
from loguru import logger
from weights_auto_tune_pipeline.logging_ import setup_logging
from weights_auto_tune_pipeline.target_config import (
    TargetConfig,
    DEFAULT_TARGETS_CONFIG,
)

setup_logging()


class Gauc(Metric):
    def create_labels(
        self,
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

    @override
    def calculate_metric_for_target(
        self,
        pool_cache: pl.DataFrame,
        target_name: str,
        session_col_name: str,
        score_col_name: str,
        # nav_screen: str = "direct:tab:video_for_you:similar",
        # platform: str = "android",
        # formula_path: str = "fstorage:vk_video_266_1769078359_f",
        # metric: MetricNames = MetricNames.GAUC,
        # nav_screen_col_name: str = "navScreen",
        # platform_col_name: str = "platform",
        # formula_path_col_name: str = "formulaPath",
        # events_col_name: str = "events",
    ) -> float:
        """Computes metric."""

        label_col = f"label_{target_name}"
        if label_col not in pool_cache.columns:
            raise ValueError(f"Label {label_col!r} not found ...")

        logger.info(
            f"Run calculating {self.get_metric_name()} for target event: {target_name}"
        )
        group_stats = pool_cache.group_by(session_col_name).agg(
            [
                pl.col(label_col).sum().alias("pos_count"),
                pl.col(label_col).count().alias("total_count"),
                pl.col(label_col).min().alias("min_label"),
                pl.col(label_col).max().alias("max_label"),
            ]
        )[session_col_name]

        valid_groups = group_stats.filter(
            (pl.col("min_label") == 0) & (pl.col("max_label") == 1)
        )

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

            y_true = group_pool_cache[label_col].to_numpy()
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

        # pool_cache: pl.DataFrame = pl.read_ndjson(self.path_to_pool_cache)
