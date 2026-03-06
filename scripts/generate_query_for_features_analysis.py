from pathlib import Path


def main():
    _query = (
        f"""
USE jupiter;

PRAGMA ClassicDivision = "false";
PRAGMA yt.Pool = {POOL!r};
PRAGMA yt.DefaultOperationWeight = {DEFAULT_OPERATION_WEIGHT!r};
PRAGMA AnsiInForEmptyOrNullableItemsCollections;

-- Constants
$TYPE_ID = 1776;
$USER_TYPE = "vk";
$RECOMMENDER_ID = 200;
$NAVSCREEN = "video_for_you";
$FORMULA_PATH = "{FORMULA_PATH}";
"""
    ).strip()

    features_for_select_block = ",\n".join(
        f"\t\tfeatures[{idx}] AS feature_value_{idx}" for idx in FEATURE_IDS
    )

    _query += f"""
\n$pool_cache_with_extracted_features =
    SELECT
        TablePath() AS table_path,
{features_for_select_block}
    FROM RANGE(
        `{PATH_TO_POOL_CACHE}`,
        "{START_DATE}",
        "{END_DATE}"
    )
    WHERE
        typeId == $TYPE_ID
        AND userType == $USER_TYPE
        AND recommenderId == $RECOMMENDER_ID
        AND navScreen == $NAVSCREEN
        AND platform IN ("android", "vk_video_android")
        AND formulaPath == $FORMULA_PATH;
    """

    _for_union = []
    for idx in FEATURE_IDS:
        _for_union.append(
            f"\tSELECT table_path, 'feature_{idx}' AS feature_name, feature_value_{idx} AS feature_value "
            f"FROM {POOL_CACHE_WITH_EXTRACTED_FEATURES_TABLE_NAME} WHERE feature_value_{idx} IS NOT NULL"
        )

    for_union = "\n\tUNION ALL\n".join(_for_union)
    _query += f"""
$unpivoted = (
{for_union}
);
"""

    _query += f"""
SELECT
  table_path,
  feature_name,
  PERCENTILE(feature_value, 0.1) AS p10,
  PERCENTILE(feature_value, 0.2) AS p20,
  PERCENTILE(feature_value, 0.3) AS p30,
  PERCENTILE(feature_value, 0.4) AS p40,
  PERCENTILE(feature_value, 0.5) AS p50,
  PERCENTILE(feature_value, 0.6) AS p60,
  PERCENTILE(feature_value, 0.7) AS p70,
  PERCENTILE(feature_value, 0.8) AS p80,
  PERCENTILE(feature_value, 0.9) AS p90
FROM {UNPIVOTED_TABLE_NAME}
GROUP BY table_path, feature_name
;
"""

    with (
        Path.cwd().joinpath(PATH_TO_OUTPUT_QUERY).open(encoding="utf-8", mode="w") as f
    ):
        f.write(_query)

    if Path.cwd().joinpath(PATH_TO_OUTPUT_QUERY).exists():
        print(f"File {PATH_TO_OUTPUT_QUERY} was recorded successfully!")


if __name__ == "__main__":
    START_DATE = "2026-03-03"
    END_DATE = "2026-03-04"
    DEFAULT_OPERATION_WEIGHT = "1000"
    POOL = "ucp-vkvideo-research"
    FORMULA_PATH = "fstorage:vk_video_266_1769078359_f"
    PATH_TO_POOL_CACHE = "//home/hc/ucp/vk_video/pool_caches/1d/"
    PATH_TO_OUTPUT_QUERY = "./query.sql"
    POOL_CACHE_WITH_EXTRACTED_FEATURES_TABLE_NAME = (
        "$pool_cache_with_extracted_features"
    )
    UNPIVOTED_TABLE_NAME = "$unpivoted"
    # Get idxs from CatboostPoolBuilder Recipe
    _features_ids = "228,229,891,892,893,3234,3235,3953,3954,3678,3787,3788,3638,3639,3640,3170,3400,3641,3492,3512,3588,14608,14609,14610,3513,3514,3589,3519,3520,3592,3171,3404,3593,3521,3522,3594,14626,14627,14628,3596,3642,3643,14629,14630,14631,3597,3644,3645,14632,14633,14634,3598,3600,3602,3599,3601,3603,14635,14636,14637,14638,14639,14640,3604,3646,3647,14641,14642,14643,3605,3607,3609,3606,3608,3610,14644,14645,14646,14647,14648,14649,3611,3648,3649,14650,14651,14652,3612,3614,3616,3613,3615,3617,14653,14654,14655,14656,14657,14658,3820,3821,3824,3825,3826,3837,11297,11298,11299,14605,14606,14607,11300,11301,11302,11303,11304,11305,14664,14665,14666,14667,14668,14669,11306,11307,11308,11309,11310,11311,11312,11313,11314,14676,14677,14678,11323,11324,11325,11326,3766,3767,3768,3227,14904,14712,3527,14710,3528,14713,3529,14715,14716,14717,14718,14948,14950,14952,14954,14898,14901,14899,14900,14902,14903,1,2,3537,3538,3539,3540,5331,5332,5333,5334,5335,5336,14719,14720,14721,14722,14723,14724,11294,11295,4178,4180,14705,14706,14707,14708,4181,14944,14945,14946,14947,13415,13416,13417,14412,14413,14414,3557,3558,3559,3560,3561,3562,3563,3564,3565,3566,15029,15030,15031,15032,3681,3682,4216,15025,15026,15027,15028,15021,15022,15023,15024,8075,8076,8077,8078,8090,8091,8092,8093,15017,15018,15019,15020,3868,3869,3870,3871,5875,14915,14938,14939,14940,14831,14835,14836,14837,14838,14813,14814,14822,14823,14817,14818,14826,14827,14828,14829,14819,14824,14820,14821"
    FEATURE_IDS = _features_ids.split(",")

    main()
