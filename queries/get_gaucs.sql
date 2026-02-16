-- === COMPUTE GAUC ===
use jupiter;

PRAGMA yt.DefaultOperationWeight = "1000.0";
PRAGMA yt.InferSchema = "1";
PRAGMA yt.MaxRowWeight = "128M";
PRAGMA yt.StaticPool = "ucp-vkvideo-pool-cache";

-- Constants
$TYPE_ID = 1776;
$USER_TYPE = "vk";
$RECOMMENDER_ID = 200;

-- Vars
$start_date = "2026-02-10";
$end_date = $start_date;
$path_to_pool_cache = "//home/hc/ucp/vk_video/pool_caches/1d/";

-- AUX functions
$compute_auc = ($sum_ranks, $n_pos, $n_total) -> {
    RETURN IF(
        $n_pos > 0 AND $n_pos < $n_total,
        1.0 * ($sum_ranks - $n_pos * ($n_pos + 1) / 2) / ($n_pos * ($n_total - $n_pos)),
        NULL
    );
};

DEFINE ACTION $get_gauc($formulaPath, $watchCoverageThreshold) AS

$metrics = (
    SELECT
        rid,
        CAST(viewTimeSec >= $watchCoverageThreshold AS int) AS watch_label,
        CAST(ListHas(events, "actionLike") AND NOT ListHas(events, "actionUnlike") AS int) AS like_label,
        1 - CAST(ListHas(events, "actionDislike") AND NOT ListHas(events, "actionUndislike") AS int) AS dislike_label,
        score
    FROM RANGE(
        $path_to_pool_cache,
        $start_date,
        $end_date
    )
    WHERE
        typeId == $TYPE_ID
        AND userType == $USER_TYPE
        AND recommenderId == $RECOMMENDER_ID
        AND formulaPath == $formulaPath
        AND navScreen == "video_for_you"
        AND platform IN ("android", "vk_video_android")
);

$total_unique_sessions_tbl = (
    SELECT COUNT(DISTINCT rid) AS total_unique_sessions
    FROM $metrics
);

$ranked = (
    SELECT
        rid,
        watch_label,
        like_label,
        dislike_label,
        score,
        ROW_NUMBER() OVER (PARTITION BY rid ORDER BY score ASC) - 1 AS rang
    FROM $metrics
);

$session_stats = (
    SELECT
        rid,
        COUNT(*) AS n_total,

        SUM(watch_label) AS watch_n_pos,
        SUM(IF(watch_label == 1, rang + 1, 0)) AS watch_sum_ranks,

        SUM(like_label) AS like_n_pos,
        SUM(IF(like_label == 1, rang + 1, 0)) AS like_sum_ranks,

        SUM(dislike_label) AS dislike_n_pos,
        SUM(IF(dislike_label == 1, rang + 1, 0)) AS dislike_sum_ranks
    FROM $ranked
    GROUP BY rid
);

$empty_sessions_tbl = (
    SELECT
        COUNT(*) AS empty_sessions,
        SUM(n_total) AS empty_samples
    FROM $session_stats
    WHERE n_total < 2
);

$watch_valid = (
    SELECT
        n_total,
        $compute_auc(watch_sum_ranks, watch_n_pos, n_total) AS auc
    FROM $session_stats
    WHERE n_total >= 2 AND watch_n_pos > 0 AND watch_n_pos < n_total
);

$like_valid = (
    SELECT
        n_total,
        $compute_auc(like_sum_ranks, like_n_pos, n_total) AS auc
    FROM $session_stats
    WHERE n_total >= 2 AND like_n_pos > 0 AND like_n_pos < n_total
);

$dislike_valid = (
    SELECT
        n_total,
        $compute_auc(dislike_sum_ranks, dislike_n_pos, n_total) AS auc
    FROM $session_stats
    WHERE n_total >= 2 AND dislike_n_pos > 0 AND dislike_n_pos < n_total
);

$watch_metrics_tbl = (
    SELECT 'GAUCSimple_watch_coverage' AS metric, AVG(auc) AS value FROM $watch_valid
    UNION ALL
    SELECT 'GAUCWeighted_watch_coverage' AS metric, (SUM(auc * n_total) / SUM(n_total)) AS value FROM $watch_valid
    UNION ALL
    SELECT 'watch_sessions' AS metric, COUNT(*) AS value FROM $watch_valid
    UNION ALL
    SELECT 'watch_samples' AS metric, SUM(n_total) AS value FROM $watch_valid
);

$like_metrics_tbl = (
    SELECT 'GAUCSimple_like' AS metric, AVG(auc) AS value FROM $like_valid
    UNION ALL
    SELECT 'GAUCWeighted_like' AS metric, (SUM(auc * n_total) / SUM(n_total)) AS value FROM $like_valid
    UNION ALL
    SELECT 'like_sessions' AS metric, COUNT(*) AS value FROM $like_valid
    UNION ALL
    SELECT 'like_samples' AS metric, SUM(n_total) AS value FROM $like_valid
);

$dislike_metrics_tbl = (
    SELECT 'GAUCSimple_dislike' AS metric, AVG(auc) AS value FROM $dislike_valid
    UNION ALL
    SELECT 'GAUCWeighted_dislike' AS metric, (SUM(auc * n_total) / SUM(n_total)) AS value FROM $dislike_valid
    UNION ALL
    SELECT 'dislike_sessions' AS metric, COUNT(*) AS value FROM $dislike_valid
    UNION ALL
    SELECT 'dislike_samples' AS metric, SUM(n_total) AS value FROM $dislike_valid
);

-- Общая статистика
$total_sessions_tbl = (
    SELECT 'total_sessions' AS metric, COUNT(*) AS value FROM $session_stats
);

$total_samples_tbl = (
    SELECT 'total_samples' AS metric, SUM(n_total) AS value FROM $session_stats
);

$total_unique_sessions_val = (
    SELECT 'total_unique_sessions_raw' AS metric, total_unique_sessions AS value FROM $total_unique_sessions_tbl
);

$empty_sessions_val = (
    SELECT 'empty_sessions' AS metric, empty_sessions AS value FROM $empty_sessions_tbl
);

$empty_samples_val = (
    SELECT 'empty_samples' AS metric, empty_samples AS value FROM $empty_sessions_tbl
);

$empty_pct = (
    SELECT
        'empty_sessions_pct' AS metric,
        IF(
            u.total_unique_sessions > 0,
            100.0 * e.empty_sessions / u.total_unique_sessions,
            0.0
        ) AS value
    FROM $empty_sessions_tbl e
    CROSS JOIN $total_unique_sessions_tbl u
);

-- Объединяем всё
$all_metrics = (
    SELECT * FROM $watch_metrics_tbl
    UNION ALL
    SELECT * FROM $like_metrics_tbl
    UNION ALL
    SELECT * FROM $dislike_metrics_tbl
    UNION ALL
    SELECT * FROM $total_sessions_tbl
    UNION ALL
    SELECT * FROM $total_samples_tbl
    UNION ALL
    SELECT * FROM $total_unique_sessions_val
    UNION ALL
    SELECT * FROM $empty_sessions_val
    UNION ALL
    SELECT * FROM $empty_samples_val
    UNION ALL
    SELECT * FROM $empty_pct
);

SELECT
    $formulaPath AS formulaPath,
    $start_date AS startDate,
    $watchCoverageThreshold AS watchCoverageThreshold,
    t.*
FROM $all_metrics AS t
ORDER BY metric;

END DEFINE;

-- == START == --
-- watchCoverageRecord = 30
DO $get_gauc("fstorage:vk_video_266_1769078359_f", 30);
DO $get_gauc("fstorage:vk_video_310_1770883950_k", 30);
DO $get_gauc("fstorage:vk_video_310_1770880753_v", 30);
DO $get_gauc("fstorage:vk_video_310_1770800906_w", 30);
DO $get_gauc("fstorage:vk_video_266_1770751611_n", 30);
DO $get_gauc("fstorage:vk_video_266_1770737548_e", 30);
DO $get_gauc("fstorage:vk_video_266_1770725758_w", 30);
DO $get_gauc("fstorage:vk_video_310_1770724317_p", 30);
DO $get_gauc("fstorage:vk_video_282_1770382670_v", 30);

-- watchCoverageRecord = 900
DO $get_gauc("fstorage:vk_video_266_1769078359_f", 900);
DO $get_gauc("fstorage:vk_video_310_1770883950_k", 900);
DO $get_gauc("fstorage:vk_video_310_1770880753_v", 900);
DO $get_gauc("fstorage:vk_video_310_1770800906_w", 900);
DO $get_gauc("fstorage:vk_video_266_1770751611_n", 900);
DO $get_gauc("fstorage:vk_video_266_1770737548_e", 900);
DO $get_gauc("fstorage:vk_video_266_1770725758_w", 900);
DO $get_gauc("fstorage:vk_video_310_1770724317_p", 900);
DO $get_gauc("fstorage:vk_video_282_1770382670_v", 900);
