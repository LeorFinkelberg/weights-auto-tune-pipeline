-- === COMPUTE GAUC WITH TEST IDS ===
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
$start_date = "2026-02-15";
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

$extract_test_ids = ($requestContext) -> {
    RETURN Yson::ConvertToStringList(
        Yson::Lookup($requestContext, "testIds")
    );
};

DEFINE ACTION $get_gauc($testId, $formulaPath, $watchCoverageThreshold) AS

$metrics_with_tests = (
    SELECT
        rid,
        $extract_test_ids(requestContext) AS test_ids,
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
        AND navScreen == "video_for_you"
        AND formulaPath == $formulaPath
        -- AND platform IN ("android", "vk_video_android")
);

$metrics_exploded = (
    SELECT
        rid,
        test_id,
        watch_label,
        like_label,
        dislike_label,
        score
    FROM $metrics_with_tests
    FLATTEN LIST BY test_ids AS test_id
    WHERE test_id IS NOT NULL
);

$total_unique_sessions_tbl = (
    SELECT COUNT(DISTINCT rid) AS total_unique_sessions
    FROM $metrics_with_tests
);

$ranked = (
    SELECT
        rid,
        test_id,
        watch_label,
        like_label,
        dislike_label,
        score,
        ROW_NUMBER() OVER (PARTITION BY rid, test_id ORDER BY score ASC) - 1 AS rang
    FROM $metrics_exploded
);

$session_stats = (
    SELECT
        rid,
        test_id,
        COUNT(*) AS n_total,

        SUM(watch_label) AS watch_n_pos,
        SUM(IF(watch_label == 1, rang + 1, 0)) AS watch_sum_ranks,

        SUM(like_label) AS like_n_pos,
        SUM(IF(like_label == 1, rang + 1, 0)) AS like_sum_ranks,

        SUM(dislike_label) AS dislike_n_pos,
        SUM(IF(dislike_label == 1, rang + 1, 0)) AS dislike_sum_ranks
    FROM $ranked
    GROUP BY rid, test_id
    -- HAVING COUNT(*) >= 2
);

$empty_sessions_tbl = (
    SELECT
        test_id,
        COUNT(*) AS empty_sessions,
        SUM(n_total) AS empty_samples
    FROM $session_stats
    WHERE n_total < 2
    GROUP BY test_id
);

$watch_valid = (
    SELECT
        test_id,
        n_total,
        $compute_auc(watch_sum_ranks, watch_n_pos, n_total) AS auc
    FROM $session_stats
    WHERE n_total >= 2 AND watch_n_pos > 0 AND watch_n_pos < n_total
);

$like_valid = (
    SELECT
        test_id,
        n_total,
        $compute_auc(like_sum_ranks, like_n_pos, n_total) AS auc
    FROM $session_stats
    WHERE n_total >= 2 AND like_n_pos > 0 AND like_n_pos < n_total
);

$dislike_valid = (
    SELECT
        test_id,
        n_total,
        $compute_auc(dislike_sum_ranks, dislike_n_pos, n_total) AS auc
    FROM $session_stats
    WHERE n_total >= 2 AND dislike_n_pos > 0 AND dislike_n_pos < n_total
);

$watch_metrics_tbl = (
    SELECT
        test_id,
        'GAUCSimple_watch_coverage' AS metric,
        AVG(auc) AS value
    FROM $watch_valid
    GROUP BY test_id

    UNION ALL

    SELECT
        test_id,
        'GAUCWeighted_watch_coverage' AS metric,
        (SUM(auc * n_total) / SUM(n_total)) AS value
    FROM $watch_valid
    GROUP BY test_id

    UNION ALL

    SELECT
        test_id,
        'watch_sessions' AS metric,
        COUNT(*) AS value
    FROM $watch_valid
    GROUP BY test_id

    UNION ALL

    SELECT
        test_id,
        'watch_samples' AS metric,
        SUM(n_total) AS value
    FROM $watch_valid
    GROUP BY test_id
);

$like_metrics_tbl = (
    SELECT
        test_id,
        'GAUCSimple_like' AS metric,
        AVG(auc) AS value
    FROM $like_valid
    GROUP BY test_id

    UNION ALL

    SELECT
        test_id,
        'GAUCWeighted_like' AS metric,
        (SUM(auc * n_total) / SUM(n_total)) AS value
    FROM $like_valid
    GROUP BY test_id

    UNION ALL

    SELECT
        test_id,
        'like_sessions' AS metric,
        COUNT(*) AS value
    FROM $like_valid
    GROUP BY test_id

    UNION ALL

    SELECT
        test_id,
        'like_samples' AS metric,
        SUM(n_total) AS value
    FROM $like_valid
    GROUP BY test_id
);

$dislike_metrics_tbl = (
    SELECT
        test_id,
        'GAUCSimple_dislike' AS metric,
        AVG(auc) AS value
    FROM $dislike_valid
    GROUP BY test_id

    UNION ALL

    SELECT
        test_id,
        'GAUCWeighted_dislike' AS metric,
        (SUM(auc * n_total) / SUM(n_total)) AS value
    FROM $dislike_valid
    GROUP BY test_id

    UNION ALL

    SELECT
        test_id,
        'dislike_sessions' AS metric,
        COUNT(*) AS value
    FROM $dislike_valid
    GROUP BY test_id

    UNION ALL

    SELECT
        test_id,
        'dislike_samples' AS metric,
        SUM(n_total) AS value
    FROM $dislike_valid
    GROUP BY test_id
);

$total_sessions_tbl = (
    SELECT
        test_id,
        'total_sessions' AS metric,
        COUNT(*) AS value
    FROM $session_stats
    GROUP BY test_id
);

$total_samples_tbl = (
    SELECT
        test_id,
        'total_samples' AS metric,
        SUM(n_total) AS value
    FROM $session_stats
    GROUP BY test_id
);

$empty_sessions_val = (
    SELECT
        test_id,
        'empty_sessions' AS metric,
        empty_sessions AS value
    FROM $empty_sessions_tbl
);

$empty_samples_val = (
    SELECT
        test_id,
        'empty_samples' AS metric,
        empty_samples AS value
    FROM $empty_sessions_tbl
);

$empty_pct = (
    SELECT
        e.test_id,
        'empty_sessions_pct' AS metric,
        (100.0 * e.empty_sessions / u.total_sessions) AS value
    FROM $empty_sessions_tbl e
    LEFT JOIN (
        SELECT test_id, COUNT(*) AS total_sessions
        FROM $session_stats
        GROUP BY test_id
    ) u ON e.test_id == u.test_id
);

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
    SELECT * FROM $empty_sessions_val
    UNION ALL
    SELECT * FROM $empty_samples_val
    UNION ALL
    SELECT * FROM $empty_pct
);

SELECT
    test_id,
    metric,
    value
FROM $all_metrics
WHERE test_id == $testId
ORDER BY test_id, metric;

END DEFINE;

-- == START == --
-- testId, formulaPath, watchCoverageThreshold
DO $get_gauc("10029670", "fstorage:vk_video_266_1769078359_f", 900);
DO $get_gauc("10029671", "fstorage:vk_video_266_1770725758_w", 900);
DO $get_gauc("10029672", "fstorage:vk_video_266_1770737548_e", 900);
