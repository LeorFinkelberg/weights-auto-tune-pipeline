import warnings
import requests
import polars as pl
import numpy as np
import urllib3

from typing import Dict, List
from loguru import logger
from pathlib import Path
from auto_tune_weights_pipeline.types_ import StrPath
from auto_tune_weights_pipeline.constants import (
    BIG_NEGATIVE_DEFAULT_VALUE,
    DICTIONARY_HUB_URL,
    DICTIONARY_PROJECT_NAME_RECOMMENDER_UCP_VIDEO_AND_CLIPS,
)


def _disable_warnings():
    warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


_disable_warnings()


class FeaturesPairsGenerator:
    def __init__(
        self,
        path_to_feature_names: StrPath,
        type_id: int = 1776,
        user_type: str = "vk",
        recommender_id: int = 200,
        dictionary_hub_url: str = DICTIONARY_HUB_URL,
        dictionary_hub_project: str = DICTIONARY_PROJECT_NAME_RECOMMENDER_UCP_VIDEO_AND_CLIPS,
    ) -> None:
        self.type_id = type_id
        self.user_type = user_type
        self.recommender_id = recommender_id
        self.dictionary_hub_url = dictionary_hub_url
        self.dictionary_hub_project = dictionary_hub_project
        self.features: List[int] = self._map_feature_names_to_feature_ids(
            path_to_feature_names
        )
        self.feature_to_idx: Dict[int, int] = {
            fid: i for i, fid in enumerate(self.features)
        }

    def _map_feature_names_to_feature_ids(
        self,
        path_to_feature_names: StrPath,
        n: int = 30,
    ) -> List[int]:
        feature_name_to_feature_id: Dict[str, int] = requests.post(
            url=self.dictionary_hub_url,
            json=[self.dictionary_hub_project],
            verify=False,
        ).json()["name2idMap"][self.dictionary_hub_project]["features"]

        features: List[str] = []
        with (
            Path.cwd()
            .joinpath(path_to_feature_names)
            .open(encoding="utf-8") as _features
        ):
            for line in _features:
                features.append(line.strip())

        feature_ids: List[int] = []
        for feature_name in features:
            try:
                feature_ids.append(feature_name_to_feature_id[feature_name])
            except KeyError:
                continue

        _first_n_features = ",".join(str(id_) for id_ in feature_ids[:n])
        _last_n_features = ",".join(str(id_) for id_ in feature_ids[-n:])

        logger.info(
            "Feature ids: {} ... {}".format(_first_n_features, _last_n_features)
        )
        logger.info("Features count: {}".format(len(feature_ids)))

        return feature_ids

    @staticmethod
    def _consumption_time_batch_impl(
        view_times: pl.Series, durations: pl.Series
    ) -> pl.Series:
        v = view_times.to_numpy().astype(np.float64)
        d = durations.to_numpy().astype(np.float64)

        result = np.zeros_like(v)
        mask = v > 0.0

        if not np.any(mask):
            return pl.Series(result)

        v_masked = v[mask]
        d_masked = d[mask]

        magic_duration = (
            1.27494682e-05 * d_masked * d_masked
            + 2.29489044e-02 * d_masked
            + 3.18074092e01
        )
        nwt = np.where(
            d_masked > 7200.0, 850.0, np.minimum(magic_duration, d_masked / 2.0)
        )

        click_weight = 0.01 * nwt
        regain_coeff = d_masked / (d_masked - nwt + click_weight)
        nw_time = regain_coeff * np.maximum(v_masked - nwt, click_weight)

        smooth = 0.25
        U = 7200.0
        C = 600.0
        this_clamp = np.minimum(nw_time, U)
        max_clamp = np.minimum(d_masked, U)
        target_regain = np.power(U, 1.0 - smooth)
        max_target = U / C

        result[mask] = (
            this_clamp
            * max_target
            / ((np.power(max_clamp, smooth) + 1e-7) * target_regain)
        )

        return pl.Series(result)

    def consumption_time_batch(
        self, view_times: pl.Expr, durations: pl.Expr
    ) -> pl.Expr:
        return pl.struct(
            [view_times.alias("viewTimeSec"), durations.alias("durationSeconds")]
        ).map_batches(
            lambda s: self._consumption_time_batch_impl(
                s.struct.field("viewTimeSec"), s.struct.field("durationSeconds")
            ),
            return_dtype=pl.Float64,
        )

    def extract_features_dict(self, features_list: list) -> dict[int, float]:
        result = {fid: BIG_NEGATIVE_DEFAULT_VALUE for fid in self.features}

        if not features_list:
            return result

        for item in features_list:
            try:
                fid = int(item[0])
                value = item[1]
                if fid in self.feature_to_idx and value is not None:
                    result[fid] = round(float(value), 8)
            except (ValueError, TypeError, IndexError):
                continue

        return result

    def _extract_features_batch(self, features_series: pl.Series) -> np.ndarray:
        features_list = features_series.to_list()
        n_rows = len(features_list)
        n_features = len(self.features)

        result_matrix = np.full(
            (n_rows, n_features), BIG_NEGATIVE_DEFAULT_VALUE, dtype=np.float32
        )

        for i, features in enumerate(features_list):
            if features:
                for feat in features:
                    try:
                        fid = int(feat[0])
                        if fid in self.feature_to_idx:
                            idx = self.feature_to_idx[fid]
                            value = float(feat[1])
                            if not np.isnan(value):
                                result_matrix[i, idx] = round(value, 8)
                    except (ValueError, TypeError, IndexError):
                        continue

        return result_matrix

    def generate_features_table(
        self,
        df: pl.DataFrame,
        like_weight: float,
        dislike_weight: float,
        consumption_time_weight: float,
    ) -> pl.DataFrame:
        logger.info(f"Feature table creating with {len(df)} rows ...")

        filtered_df = df.filter(
            (pl.col("typeId") == self.type_id)
            & (pl.col("userType") == self.user_type)
            & (pl.col("recommenderId") == self.recommender_id)
            & pl.col("events")
            .list.eval(
                pl.element().is_in(
                    ["actionDislike", "actionLike", "watchCoverageRecord", "card_view"]
                )
            )
            .list.any()
        )

        if len(filtered_df) == 0:
            return pl.DataFrame()

        filtered_df = filtered_df.with_columns(
            [
                # like target
                (
                    pl.when(
                        pl.col("events").list.contains("actionLike")
                        & ~pl.col("events").list.contains("actionUnlike")
                    )
                    .then(1.0)
                    .otherwise(0.0)
                ).alias("like_target"),
                # dislike target
                (
                    pl.when(
                        pl.col("events").list.contains("actionDislike")
                        & ~pl.col("events").list.contains("actionUndislike")
                    )
                    .then(0.0)
                    .otherwise(1.0)
                ).alias("dislike_target"),
                # consumption_time target
                self.consumption_time_batch(
                    pl.col("viewTimeSec"), pl.col("durationSeconds")
                ).alias("consumption_time_target"),
            ]
        )

        feature_matrix = self._extract_features_batch(filtered_df["features"])

        n_features = len(self.features)
        feature_cols = [
            pl.Series(
                name=f"f_{i}",
                values=[
                    f"{val:.8g}".rstrip("0").rstrip(".") for val in feature_matrix[:, i]
                ],
            )
            for i in range(n_features)
        ]

        filtered_df = filtered_df.with_columns(feature_cols)
        filtered_df = filtered_df.with_columns(
            (
                2.0
                * (2.0 * (2.0 * 0.0 + pl.col("like_target")) + pl.col("dislike_target"))
                + pl.col("consumption_time_target")
            ).alias("label_value")
        )

        n_features = len(self.features)
        value_expr = pl.format(
            "{}\t1\t{}",
            pl.col("label_value")
            .cast(pl.String)
            .str.replace(pl.lit(r"[.]?0+$"), pl.lit("")),
            pl.concat_list([pl.col(f"f_{i}") for i in range(n_features)]).list.join(
                pl.lit("\t")
            ),
        )

        filtered_df = filtered_df.with_columns(value_expr.alias("value"))

        targets_expr = pl.concat_list(
            [
                pl.struct(
                    [
                        pl.lit("like").alias("name"),
                        pl.col("like_target").alias("target"),
                        pl.lit(like_weight).alias("weight"),
                    ]
                ),
                pl.struct(
                    [
                        pl.lit("dislike").alias("name"),
                        pl.col("dislike_target").alias("target"),
                        pl.lit(dislike_weight).alias("weight"),
                    ]
                ),
                pl.struct(
                    [
                        pl.lit("consumption_time").alias("name"),
                        pl.col("consumption_time_target").alias("target"),
                        pl.lit(consumption_time_weight).alias("weight"),
                    ]
                ),
            ]
        ).alias("targets")

        filtered_df = filtered_df.with_columns(targets_expr)

        result_df = filtered_df.with_columns([pl.col("rid").alias("key")]).select(
            ["key", "value", "targets"]
        )

        result_df = (
            result_df.with_columns(
                pl.col("targets")
                .list.eval(pl.element().struct.field("target"))
                .list.unique()
                .list.len()
                .alias("uniqueTargetsCount")
            )
            .filter(pl.col("uniqueTargetsCount") > 1)
            .drop("uniqueTargetsCount")
        )

        result_df = result_df.sort("key")

        logger.info(f"Feature table: {len(result_df)} rows")

        result_df = result_df.select(["key", "value", "targets"])

        return result_df

    @staticmethod
    def generate_pairs_table(features_df: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"Pair generating with {len(features_df)} rows of features ...")

        if len(features_df) == 0:
            return pl.DataFrame()

        features_df = features_df.with_columns(
            pl.int_range(0, pl.len()).alias("rowIndex")
        )

        exploded_df = features_df.explode("targets").with_columns(
            [
                pl.col("targets").struct.field("name").alias("targetName"),
                pl.col("targets").struct.field("target").alias("target"),
                pl.col("targets").struct.field("weight").alias("weight"),
                pl.col("key").alias("groupId"),
            ]
        )

        logger.info("Data grouping ...")

        grouped = exploded_df.group_by(["groupId", "targetName"]).agg(
            [
                pl.col("rowIndex").alias("row_indices"),
                pl.col("target").alias("targets_list"),
                pl.col("weight").first().alias("weight"),
                pl.len().alias("groupSize"),
            ]
        )

        logger.info("Pairs creating ...")

        expanded = []
        for group in grouped.iter_rows(named=True):
            group_id = group["groupId"]
            target_name = group["targetName"]
            weight = group["weight"]
            row_indices = group["row_indices"]
            targets = group["targets_list"]
            group_size = group["groupSize"]

            group_df = pl.DataFrame(
                {
                    "idx": row_indices,
                    "target": targets,
                    "groupId": [group_id] * group_size,
                    "targetName": [target_name] * group_size,
                    "weight": [weight] * group_size,
                }
            )
            expanded.append(group_df)

        if not expanded:
            logger.warning("Has not pairs for generating ...")
            return pl.DataFrame()

        all_groups_df = pl.concat(expanded)

        pairs_df = (
            all_groups_df.join(
                all_groups_df.select(["idx", "target", "groupId", "targetName"]),
                on=["groupId", "targetName"],
                suffix="_2",
            )
            .filter(pl.col("idx") < pl.col("idx_2"))
            .select(
                [
                    pl.col("idx").alias("idx1"),
                    pl.col("target").alias("target1"),
                    pl.col("idx_2").alias("idx2"),
                    pl.col("target_2").alias("target2"),
                    "targetName",
                    "weight",
                    "groupId",
                ]
            )
        )

        logger.info(f"Unordered pairs: {len(pairs_df)}")
        total_weights = pairs_df.group_by(["groupId", "targetName"]).agg(
            (pl.col("target1") - pl.col("target2")).abs().sum().alias("totalRawWeight")
        )

        result_df = (
            pairs_df.join(total_weights, on=["groupId", "targetName"])
            .with_columns(
                [
                    pl.when(pl.col("target1") > pl.col("target2"))
                    .then(pl.col("idx1"))
                    .otherwise(pl.col("idx2"))
                    .alias("winner"),
                    pl.when(pl.col("target1") > pl.col("target2"))
                    .then(pl.col("idx2"))
                    .otherwise(pl.col("idx1"))
                    .alias("looser"),
                    (
                        pl.col("weight") * (pl.col("target1") - pl.col("target2")).abs()
                    ).alias("pair_weight"),
                ]
            )
            .filter(pl.col("pair_weight").abs() >= 1e-8)
        )

        final_df = result_df.select(
            [
                pl.col("winner").cast(pl.Utf8).alias("key"),
                pl.concat_str(
                    [
                        pl.col("looser").cast(pl.Utf8),
                        pl.col("pair_weight").cast(pl.Utf8),
                    ],
                    separator="\t",
                ).alias("value"),
                pl.col("targetName").alias("target"),
            ]
        )

        logger.info(f"Generated {len(final_df)} pairs ...")
        return final_df
