import warnings

import requests
import polars as pl
import math
import urllib3

from loguru import logger
from pathlib import Path
from auto_tune_weights_pipeline.types_ import StrPath
from auto_tune_weights_pipeline.constants import (
    DICTIONARY_HUB_URL,
    DICTIONARY_PROJECT_NAME_RECOMMENDER_UCP_VIDEO_AND_CLIPS,
)


def _disable_warnings():
    warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


_disable_warnings()


def _disable_warnings(self):
    warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
        self.features: list[int] = self._map_feature_names_to_feature_ids(
            path_to_feature_names
        )

    def _map_feature_names_to_feature_ids(
        self,
        path_to_feature_names: StrPath,
    ) -> list[int]:
        feature_name_to_feature_id: dict[str, int] = requests.post(
            url=self.dictionary_hub_url,
            json=[self.dictionary_hub_project],
            verify=False,
        ).json()["name2idMap"][self.dictionary_hub_project]["features"]

        features: list[str] = []
        with (
            Path.cwd()
            .joinpath(path_to_feature_names)
            .open(encoding="utf-8") as _features
        ):
            for line in _features:
                features.append(line.strip())

        feature_ids: list[int] = [
            feature_name_to_feature_id[feature_name] for feature_name in features
        ]

        logger.info("Feature ids: {}".format(",".join(str(id_) for id_ in feature_ids)))
        logger.info("Features count: {}".format(len(feature_ids)))

        return feature_ids

    @staticmethod
    def consumption_time(view_time_sec: float, duration: float) -> float:
        if view_time_sec <= 0.0:
            return 0.0

        magic_duration = (
            1.27494682e-05 * duration * duration
            + 2.29489044e-02 * duration
            + 3.18074092e01
        )

        if duration > 7200.0:
            noisy_watching_threshold = 850.0
        else:
            noisy_watching_threshold = min(magic_duration, duration / 2.0)

        click_weight = 0.01 * noisy_watching_threshold
        regain_coeff = duration / (duration - noisy_watching_threshold + click_weight)
        nw_time = regain_coeff * max(
            view_time_sec - noisy_watching_threshold, click_weight
        )

        smooth = 0.25
        U = 7200.0
        C = 600.0
        this_clamp = min(nw_time, U)
        max_clamp = min(duration, U)
        target_regain = math.pow(U, 1.0 - smooth)
        max_target = U / C

        return (
            this_clamp
            * max_target
            / ((math.pow(max_clamp, smooth) + 1e-7) * target_regain)
        )

    @staticmethod
    def to_label_weight(targets: list[tuple[str, float, float]]) -> tuple[str, str]:
        if len(targets) > 1:
            label = str(int(sum(2**i * t[1] for i, t in enumerate(targets))))
            weight = "1"
        else:
            label = str(targets[0][1])
            weight = str(targets[0][2])
        return label, weight

    @staticmethod
    def to_targets(
        targets: list[tuple[str, float, float]],
    ) -> list[tuple[str, float, float]]:
        if len(targets) > 1:
            return targets
        else:
            return [("single_target", targets[0][1], targets[0][2])]

    @staticmethod
    def extract_feature_value(features_dict: dict, feature_id: int) -> float:
        value = features_dict.get(feature_id)
        if value is None:
            return -3.4e38
        return round(value, 8)

    def process_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        filtered_df = df.filter(
            (pl.col("typeId") == 1776)
            & (pl.col("userType") == "vk")
            & (pl.col("recommenderId") == 200)
            & pl.col("events")
            .list.eval(
                pl.element().is_in(
                    ["actionDislike", "actionLike", "watchCoverageRecord", "card_view"]
                )
            )
            .list.any()
        )

        def extract_all_features(row: dict) -> dict:
            features_list = row["features"]
            result = {}

            features_dict = {}
            if features_list:
                for item in features_list:
                    if isinstance(item, dict):
                        key = item.get("key")
                        value = item.get("value")
                        if key is not None:
                            features_dict[int(key)] = value
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        features_dict[int(item[0])] = item[1]

            for feature_id in self.features:
                try:
                    value = float(features_dict.get(feature_id))
                except (ValueError, TypeError):
                    value = -3.4e38
                result[f"feature_{feature_id}"] = round(value, 8)

            events = row["events"]
            has_like = "actionLike" in events and "actionUnlike" not in events
            has_dislike = "actionDislike" in events and "actionUndislike" not in events

            result["like_target"] = 1.0 if has_like else 0.0
            result["dislike_target"] = 0.0 if has_dislike else 1.0
            result["consumption_time_target"] = self.consumption_time(
                row["viewTimeSec"], row["durationSeconds"]
            )

            label_bits = ""
            label_bits += "1" if result["like_target"] > 0.5 else "0"
            label_bits += "1" if result["dislike_target"] > 0.5 else "0"

            result["label"] = label_bits

            feature_values = [str(result[f"feature_{fid}"]) for fid in self.features]
            result["value"] = "\t".join([label_bits, "1", *feature_values])

            result["targets"] = [
                {"name": "like", "target": result["like_target"], "weight": 5.0},
                {"name": "dislike", "target": result["dislike_target"], "weight": 14.0},
                {
                    "name": "consumption_time",
                    "target": result["consumption_time_target"],
                    "weight": 0.41,
                },
            ]

            return result

        processed = filtered_df.with_columns(
            [
                pl.struct(["features", "events", "viewTimeSec", "durationSeconds"])
                .map_elements(
                    extract_all_features,
                    return_dtype=pl.Struct(
                        [
                            *[
                                pl.Field(f"feature_{fid}", pl.Float64)
                                for fid in self.features
                            ],
                            pl.Field("like_target", pl.Float64),
                            pl.Field("dislike_target", pl.Float64),
                            pl.Field("consumption_time_target", pl.Float64),
                            pl.Field("label", pl.Utf8),
                            pl.Field("value", pl.Utf8),
                            pl.Field(
                                "targets",
                                pl.List(
                                    pl.Struct(
                                        [
                                            pl.Field("name", pl.Utf8),
                                            pl.Field("target", pl.Float64),
                                            pl.Field("weight", pl.Float64),
                                        ]
                                    )
                                ),
                            ),
                        ]
                    ),
                )
                .alias("processed")
            ]
        ).unnest("processed")

        processed = processed.rename({"rid": "key"})
        processed = processed.drop(
            [
                "features",
                "like_target",
                "dislike_target",
                "consumption_time_target",
                "label",
            ]
        )

        processed = (
            processed.with_columns(
                [
                    pl.col("targets")
                    .list.eval(pl.element().struct.field("target"))
                    .list.unique()
                    .list.len()
                    .alias("uniqueTargetsCount")
                ]
            )
            .filter(pl.col("uniqueTargetsCount") > 1)
            .drop("uniqueTargetsCount")
        )

        return processed.sort("key")

    def generate_pairs(self, features_df: pl.DataFrame) -> pl.DataFrame:
        features_df = features_df.with_columns(
            [pl.int_range(0, pl.len()).alias("rowIndex")]
        )

        exploded = features_df.explode("targets").with_columns(
            [
                pl.col("key").alias("groupId"),
                pl.col("targets").struct.field("name").alias("targetName"),
                pl.col("targets").struct.field("target").alias("target"),
                pl.col("targets").struct.field("weight").alias("weight"),
            ]
        )

        grouped = exploded.group_by(["groupId", "targetName"]).agg(
            [
                pl.col("rowIndex").alias("row_indices"),
                pl.col("target").alias("targets_list"),
                pl.col("weight").first().alias("weight"),
                pl.len().alias("groupSize"),
            ]
        )

        pairs_list = []
        for group in grouped.iter_rows(named=True):
            group_id = group["groupId"]
            target_name = group["targetName"]
            weight = group["weight"]
            row_indices = group["row_indices"]
            targets = group["targets_list"]
            group_size = group["groupSize"]

            for i in range(group_size):
                for j in range(i + 1, group_size):
                    pairs_list.append(
                        {
                            "idx1": row_indices[i],
                            "target1": targets[i],
                            "idx2": row_indices[j],
                            "target2": targets[j],
                            "targetName": target_name,
                            "weight": weight,
                            "groupId": group_id,
                            "groupSize": group_size,
                        }
                    )

        if not pairs_list:
            return pl.DataFrame()

        pairs_df = pl.DataFrame(pairs_list)

        total_weights = pairs_df.group_by(["groupId", "targetName"]).agg(
            [
                (pl.col("target1") - pl.col("target2"))
                .abs()
                .sum()
                .alias("totalRawWeight")
            ]
        )

        result = (
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
                    ).alias("weight"),
                ]
            )
            .filter(pl.col("weight").abs() >= 1e-8)
            .select(
                [
                    pl.col("winner").cast(pl.Utf8).alias("key"),
                    pl.concat_str(
                        [
                            pl.col("looser").cast(pl.Utf8),
                            pl.col("weight").cast(pl.Utf8),
                        ],
                        separator="\t",
                    ).alias("value"),
                    pl.col("targetName").alias("target"),
                ]
            )
        )

        return result
