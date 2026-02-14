import typing as t

import polars as pl
import numpy as np

from typing_extensions import override
from loguru import logger
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from auto_tune_weights_pipeline.columns import Columns
from auto_tune_weights_pipeline.constants import Platforms, NavScreens
from auto_tune_weights_pipeline.metrics.base import Metric
from auto_tune_weights_pipeline.target_config import (
    TargetConfig,
    DEFAULT_TARGETS_CONFIG,
)


class GAUC(Metric):
    """Group AUC and regular AUC metric calculator."""

    @override
    def calculate_metric(
        self,
        target_configs: t.Union[list[TargetConfig], list[str], dict[str, TargetConfig]],
        session_col_name: str = Columns.RID_COL_NAME,
        score_col_name: str = Columns.SCORE_COL_NAME,
        platforms: tuple[t.Union[str, Platforms], ...] = (Platforms.VK_VIDEO_ANDROID,),
        nav_screen: str = NavScreens.VIDEO_FOR_YOU,
        formula_path: str = "fstorage:vk_video_266_1769078359_f",
        nav_screen_col_name: str = Columns.NAV_SCREEN_COL_NAME,
        platform_col_name: str = Columns.PLATFORM_COL_NAME,
        formula_path_col_name: str = Columns.FORMULA_PATH_COL_NAME,
        calculate_regular_auc: bool = True,
    ) -> dict[str, dict[str, t.Any]]:
        """Calculates metric."""

        _mask = (
            (pl.col(nav_screen_col_name) == nav_screen)
            & pl.col(platform_col_name).is_in(platforms)
            & (pl.col(formula_path_col_name) == formula_path)
        )
        pool_cache: pl.DataFrame = pl.read_ndjson(self.path_to_pool_cache).filter(_mask)

        return self._calculate_metric_for_all_targets(
            pool_cache,
            target_configs=target_configs,
            session_col_name=session_col_name,
            score_col_name=score_col_name,
            calculate_regular_auc=calculate_regular_auc,
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
                f"label_{config.target_name}"
            )
            label_exprs.append(label_expr)

        return pool_cache.with_columns(label_exprs)

    def _calculate_metric_for_all_targets(
        self,
        pool_cache: pl.DataFrame,
        target_configs: t.Union[list[TargetConfig], list[str], dict[str, TargetConfig]],
        session_col_name: str = Columns.RID_COL_NAME,
        score_col_name: str = Columns.SCORE_COL_NAME,
        calculate_regular_auc: bool = True,
    ) -> dict[str, dict[str, t.Any]]:
        """Calculates metric for all targets."""

        pool_cache_with_labels = self._create_labels(pool_cache, target_configs)
        target_names = self._extract_target_names(target_configs)

        if not target_names:
            logger.warning("No valid target names found")
            return {}

        results = {}
        for target_name in target_names:
            result = self._calculate_both_metrics_for_target(
                pool_cache_with_labels,
                target_name,
                session_col_name,
                score_col_name,
                calculate_regular_auc=calculate_regular_auc,
            )
            results[target_name] = result

        return results

    @staticmethod
    def _extract_target_names(
        target_configs: t.Union[list[TargetConfig], list[str], dict[str, TargetConfig]],
    ) -> list[str]:
        """Extracts target names from various input formats."""

        if isinstance(target_configs, dict):
            return list(target_configs.keys())

        elif isinstance(target_configs, list):
            if not target_configs:
                logger.warning("Empty target_configs list provided")
                return []

            first_element = target_configs[0]

            if isinstance(first_element, str):
                if not all(isinstance(item, str) for item in target_configs):
                    raise TypeError(
                        "Mixed types in target_configs list. "
                        "All elements must be strings or all must be TargetConfig."
                    )
                return target_configs

            elif isinstance(first_element, TargetConfig):
                if not all(isinstance(item, TargetConfig) for item in target_configs):
                    raise TypeError(
                        "Mixed types in target_configs list. "
                        "All elements must be strings or all must be TargetConfig."
                    )
                return [config.target_name for config in target_configs]

            else:
                raise TypeError(
                    f"Unsupported element type in list: {type(first_element)}. "
                    "Expected str or TargetConfig."
                )
        else:
            raise TypeError(
                f"Unsupported target_configs type: {type(target_configs)}. "
                "Expected dict, list[str] or list[TargetConfig]."
            )

    def _calculate_both_metrics_for_target(
        self,
        pool_cache: pl.DataFrame,
        target_name: str,
        session_col_name: str,
        score_col_name: str,
        calculate_regular_auc: bool = True,
    ) -> dict[str, t.Any]:
        """Calculates both GAUC and regular AUC for a target."""

        label_col_name = f"label_{target_name}"
        logger.info(f"Calculating metrics for target: {target_name}")

        result = {
            "target": target_name,
            "total_samples": len(pool_cache),
            "total_positives": pool_cache[label_col_name].sum(),
            "positive_rate": float(pool_cache[label_col_name].mean()),
        }

        if calculate_regular_auc:
            auc_result = self._calculate_regular_auc(
                pool_cache, target_name, score_col_name
            )
            result.update(auc_result)

        gauc_result = self._calculate_gauc_for_target(
            pool_cache, target_name, session_col_name, score_col_name
        )
        result.update(gauc_result)

        return result

    @staticmethod
    def _calculate_regular_auc(
        pool_cache: pl.DataFrame,
        target_name: str,
        score_col_name: str,
    ) -> dict[str, t.Any]:
        """Calculates regular AUC."""

        label_col_name = f"label_{target_name}"

        # unique_labels = pool_cache[label_col_name].unique().to_list()
        unique_labels = pool_cache[label_col_name].unique()
        if len(unique_labels) < 2:
            logger.warning(
                f"Cannot calculate AUC for target {target_name}: "
                f"only {unique_labels} classes found"
            )
            return {
                "AUC": 0.0,
                "AUC_valid": False,
                "AUC_error": f"Insufficient classes: {unique_labels}",
            }

        try:
            y_true = np.nan_to_num(pool_cache[label_col_name].to_numpy(), nan=0.0)
            y_score = np.nan_to_num(pool_cache[score_col_name].to_numpy(), nan=0.0)

            auc = roc_auc_score(y_true, y_score)

            if np.isnan(auc):
                logger.warning(f"AUC is NaN for target {target_name}")
                return {
                    "AUC": 0.0,
                    "AUC_valid": False,
                    "AUC_error": "AUC calculation resulted in NaN",
                }

            logger.info(f"Regular AUC for {target_name}: {auc:.4f}")

            return {"AUC": float(auc), "AUC_valid": True, "AUC_error": None}
        except Exception as e:
            logger.error(f"Error calculating AUC for {target_name}: {e}")
            return {"AUC": 0.0, "AUC_valid": False, "AUC_error": str(e)}

    def _calculate_gauc_for_target(
        self,
        pool_cache: pl.DataFrame,
        target_name: str,
        session_col_name: str,
        score_col_name: str,
        chunk_size: int = 100_000,
    ) -> dict[str, t.Any]:
        """Calculates GAUC for a target."""

        label_col_name = f"label_{target_name}"

        all_auc_values = []
        all_group_sizes = []
        all_group_details = []

        unique_groups = pool_cache[session_col_name].unique().to_list()
        logger.info(f"Total unique groups for GAUC: {len(unique_groups)}")

        for i in tqdm(
            range(0, len(unique_groups), chunk_size),
            desc=f"GAUC groups processing for {target_name} ...",
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

                y_true = np.nan_to_num(group_data[label_col_name].to_numpy(), nan=0.0)
                y_score = np.nan_to_num(group_data[score_col_name].to_numpy(), nan=0.0)

                order = np.argsort(y_score)
                y_true_sorted = y_true[order]

                n_pos = np.sum(y_true_sorted)
                n_neg = len(y_true_sorted) - n_pos

                auc = roc_auc_score(y_true, y_score)

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

        if not all_auc_values:
            logger.warning(
                f"No valid groups for {self.get_metric_name()} calculation for {target_name}"
            )
            return self._empty_gauc_result()

        auc_array = np.array(all_auc_values)
        weights_array = np.array(all_group_sizes)

        result = {
            "GAUCSimple": float(np.mean(auc_array)),
            "GAUCWeighted": float(np.average(auc_array, weights=weights_array)),
            "n_groups": len(auc_array),
            "std": float(np.std(auc_array)),
            "min": float(np.min(auc_array)),
            "max": float(np.max(auc_array)),
            "median": float(np.median(auc_array)),
            "group_details": all_group_details,
            "GAUC_valid": len(auc_array) > 0,
        }

        self._log_gauc_results(target_name, result)
        return result

    @staticmethod
    def _empty_gauc_result() -> dict[str, t.Any]:
        """Returns empty GAUC result."""

        return {
            "GAUCSimple": 0.0,
            "GAUCWeighted": 0.0,
            "n_groups": 0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "group_details": [],
            "GAUC_valid": False,
        }

    @staticmethod
    def _log_gauc_results(target_name: str, result: dict[str, t.Any]) -> None:
        """Logs GAUC results."""

        logger.info(f"GAUC results for {target_name}:")
        logger.info(f"Groups with AUC: {result['n_groups']}")
        logger.info(f"GAUC (weighted): {result['GAUCWeighted']:.7f}")
        logger.info(f"GAUC (simple): {result['GAUCSimple']:.7f}")
        logger.info(f"Std AUC: {result['std']:.4f}")
        logger.info(f"Min AUC: {result['min']:.4f}")
        logger.info(f"Max AUC: {result['max']:.4f}")
        logger.info(f"Median AUC: {result['median']:.4f}")

    @staticmethod
    def compare_auc_gauc(
        auc_result: dict[str, t.Any], gauc_result: dict[str, t.Any]
    ) -> dict[str, t.Any]:
        """Compares AUC and GAUC results."""

        comparison = {
            "target": auc_result.get("target", gauc_result.get("target", "unknown")),
            "AUC": auc_result.get("AUC", 0.0),
            "GAUCWeighted": gauc_result.get("GAUCWeighted", 0.0),
            "GAUCSimple": gauc_result.get("GAUCSimple", 0.0),
            "difference_auc_gauc_weighted": (
                auc_result.get("AUC", 0.0) - gauc_result.get("GAUCWeighted", 0.0)
            ),
            "difference_auc_gauc_simple": (
                auc_result.get("AUC", 0.0) - gauc_result.get("GAUCSimple", 0.0)
            ),
            "relative_diff_auc_gauc_weighted": (
                (auc_result.get("AUC", 0.0) - gauc_result.get("GAUCWeighted", 0.0))
                / max(auc_result.get("AUC", 1e-10), 1e-10)
            ),
            "n_groups": gauc_result.get("n_groups", 0),
            "total_samples": auc_result.get("total_samples", 0),
            "AUC_valid": auc_result.get("AUC_valid", False),
            "GAUC_valid": gauc_result.get("GAUC_valid", False),
        }

        diff = abs(comparison["difference_auc_gauc_weighted"])
        if diff < 0.01:
            comparison["similarity"] = "high"
        elif diff < 0.05:
            comparison["similarity"] = "medium"
        else:
            comparison["similarity"] = "low"

        return comparison

    @staticmethod
    def get_summary(results: dict[str, dict[str, t.Any]]) -> dict[str, t.Any]:
        """Creates summary of all metrics."""

        summary = {
            "total_targets": len(results),
            "targets_with_valid_auc": 0,
            "targets_with_valid_gauc": 0,
            "average_auc": 0.0,
            "average_gauc_weighted": 0.0,
            "average_gauc_simple": 0.0,
            "target_details": {},
        }

        auc_values = []
        gauc_weighted_values = []
        gauc_simple_values = []

        for target_name, result in results.items():
            if result.get("AUC_valid", False):
                summary["targets_with_valid_auc"] += 1
                auc_values.append(result.get("AUC", 0.0))

            if result.get("GAUC_valid", False):
                summary["targets_with_valid_gauc"] += 1
                gauc_weighted_values.append(result.get("GAUCWeighted", 0.0))
                gauc_simple_values.append(result.get("GAUCSimple", 0.0))

            summary["target_details"][target_name] = {
                "AUC": result.get("AUC", 0.0),
                "AUC_valid": result.get("AUC_valid", False),
                "GAUCWeighted": result.get("GAUCWeighted", 0.0),
                "GAUCSimple": result.get("GAUCSimple", 0.0),
                "GAUC_valid": result.get("GAUC_valid", False),
                "n_groups": result.get("n_groups", 0),
                "total_samples": result.get("total_samples", 0),
                "positive_rate": result.get("positive_rate", 0.0),
            }

        if auc_values:
            summary["average_auc"] = float(np.mean(auc_values))
        if gauc_weighted_values:
            summary["average_gauc_weighted"] = float(np.mean(gauc_weighted_values))
        if gauc_simple_values:
            summary["average_gauc_simple"] = float(np.mean(gauc_simple_values))

        logger.info("=" * 50)
        logger.info("METRICS SUMMARY:")
        logger.info(f"Total targets: {summary['total_targets']}")
        logger.info(f"Targets with valid AUC: {summary['targets_with_valid_auc']}")
        logger.info(f"Targets with valid GAUC: {summary['targets_with_valid_gauc']}")
        logger.info(f"Average AUC: {summary['average_auc']:.7f}")
        logger.info(f"Average GAUC (weighted): {summary['average_gauc_weighted']:.7f}")
        logger.info(f"Average GAUC (simple): {summary['average_gauc_simple']:.7f}")
        logger.info("=" * 50)

        return summary
