-- === GET SUBSAMPLE FROM POOL-CACHE ===
use jupiter;

PRAGMA yt.DefaultOperationWeight = "10.0";
PRAGMA yt.MaxRowWeight = "128M";
PRAGMA yt.StaticPool = "ucp-vkvideo-pool-cache";

INSERT INTO `//home/.../vk_video/pool_cache_features_2026-02-01_train` WITH TRUNCATE
SELECT
    `typeId`,
    `userType`,
    `recommenderId`,
    `navScreen`,
    `features`,
    `rid`,
    `events`,
    `viewTimeSec`,
    `platform`,
    `score`,
    `formulaPath`,
    `durationSeconds`
FROM jupiter.`//home/.../vk_video/pool_caches/1d/2026-02-01`
WHERE
  navScreen = "video_for_you"
  and platform in ("vk_video_android", "android")
  and formulaPath == "fstorage:vk_video_266_1769078359_f"
  and typeId == 1776
  and userType == "vk"
  and recommenderId == 200
ORDER BY rid
LIMIT 500000


-- === COMPUTE GAUC ===
use jupiter;

PRAGMA yt.DefaultOperationWeight = "10.0";
PRAGMA yt.InferSchema = "1";
PRAGMA yt.MaxRowWeight = "128M";
PRAGMA yt.StaticPool = "ucp-vkvideo-pool-cache";

-- Constants
$TYPE_ID = 1776;
$USER_TYPE = "vk";
$RECOMMENDER_ID = 200;

-- Vars
$path_to_pool_cache = "//home/hc/ucp/vk_video/pool_caches/1d/2026-02-02";
$formulaPath = "fstorage:vk_video_266_1769078359_f";
$watchCoverageThreshold = 30;


$metrics = (
    SELECT
        rid,
        CAST(viewTimeSec >= $watchCoverageThreshold AS int) AS watch_label,
        CAST(ListHas(events, "actionLike") AND NOT ListHas(events, "actionUnlike") AS int) AS like_label,
        1 - CAST(ListHas(events, "actionDislike") AND NOT ListHas(events, "actionUndislike") AS int) AS dislike_label,
        score
    FROM $path_to_pool_cache
    WHERE
        typeId == $TYPE_ID
        AND userType == $USER_TYPE
        AND recommenderId == $RECOMMENDER_ID
        AND formulaPath == $formulaPath
        AND navScreen == "video_for_you"
        AND platform IN ("android", "vk_video_android")
);

$ranked = (
    SELECT
        rid,
        watch_label,
        like_label,
        dislike_label,
        score,
        ROW_NUMBER() OVER (PARTITION BY rid ORDER BY score ASC) - 1 AS rn
    FROM $metrics
);

$session_stats = (
    SELECT
        rid,
        COUNT(*) AS n_total,

        SUM(watch_label) AS watch_n_pos,
        SUM(IF(watch_label == 1, rn + 1, 0)) AS watch_sum_ranks,

        SUM(like_label) AS like_n_pos,
        SUM(IF(like_label == 1, rn + 1, 0)) AS like_sum_ranks,

        SUM(dislike_label) AS dislike_n_pos,
        SUM(IF(dislike_label == 1, rn + 1, 0)) AS dislike_sum_ranks
    FROM $ranked
    GROUP BY rid
    HAVING COUNT(*) >= 2
);

$watch_valid = (
    SELECT
        n_total,
        1.0 * (watch_sum_ranks - watch_n_pos * (watch_n_pos + 1) / 2) /
        (watch_n_pos * (n_total - watch_n_pos)) AS auc
    FROM $session_stats
    WHERE watch_n_pos > 0 AND watch_n_pos < n_total
);

$like_valid = (
    SELECT
        n_total,
        1.0 * (like_sum_ranks - like_n_pos * (like_n_pos + 1) / 2) /
        (like_n_pos * (n_total - like_n_pos)) AS auc
    FROM $session_stats
    WHERE like_n_pos > 0 AND like_n_pos < n_total
);

$dislike_valid = (
    SELECT
        n_total,
        1.0 * (dislike_sum_ranks - dislike_n_pos * (dislike_n_pos + 1) / 2) /
        (dislike_n_pos * (n_total - dislike_n_pos)) AS auc
    FROM $session_stats
    WHERE dislike_n_pos > 0 AND dislike_n_pos < n_total
);

$all_metrics = (
    SELECT "GAUCSimple_watch_coverage" AS metric, AVG(auc) AS value FROM $watch_valid
    UNION ALL
    SELECT "GAUCWeighted_watch_coverage" AS metric, (SUM(auc * n_total) / SUM(n_total)) AS value FROM $watch_valid
    UNION ALL
    SELECT "watch_sessions" AS metric, COUNT(*) AS value FROM $watch_valid
    UNION ALL
    SELECT "watch_samples" AS metric, SUM(n_total) AS value FROM $watch_valid

    UNION ALL

    SELECT "GAUCSimple_like_simple" AS metric, AVG(auc) AS value FROM $like_valid
    UNION ALL
    SELECT "GAUCWeighted_like_weighted" AS metric, (SUM(auc * n_total) / SUM(n_total)) AS value FROM $like_valid
    UNION ALL
    SELECT "like_sessions" AS metric, COUNT(*) AS value FROM $like_valid
    UNION ALL
    SELECT "like_samples" AS metric, SUM(n_total) AS value FROM $like_valid

    UNION ALL

    SELECT "GAUCSimple_dislike_simple" AS metric, AVG(auc) AS value FROM $dislike_valid
    UNION ALL
    SELECT "GAUCWeighted_dislike_weighted" AS metric, (SUM(auc * n_total) / SUM(n_total)) AS value FROM $dislike_valid
    UNION ALL
    SELECT "dislike_sessions" AS metric, COUNT(*) AS value FROM $dislike_valid
    UNION ALL
    SELECT "dislike_samples" AS metric, SUM(n_total) AS value FROM $dislike_valid

    UNION ALL

    SELECT "total_sessions" AS metric, COUNT(*) AS value FROM $session_stats
    UNION ALL
    SELECT "total_samples" AS metric, SUM(n_total) AS value FROM $session_stats
);

SELECT * FROM $all_metrics
ORDER BY metric;