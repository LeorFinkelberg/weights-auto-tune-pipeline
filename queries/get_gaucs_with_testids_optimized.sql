-- === COMPUTE GAUC WITH TEST IDS (OPTIMIZED) ===
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

$compute_auc = ($sum_ranks, $n_pos, $n_total) -> {
    $sum_ranks_d = CAST($sum_ranks AS double);
    $n_pos_d = CAST($n_pos AS double);
    $n_total_d = CAST($n_total AS double);

    RETURN IF(
        $n_pos > 0 AND $n_pos < $n_total,
        ($sum_ranks_d - $n_pos_d * ($n_pos_d + 1.0) / 2.0) /
        ($n_pos_d * ($n_total_d - $n_pos_d)),
        NULL
    );
};

$extract_test_ids = ($requestContext) -> {
    RETURN Yson::ConvertToStringList(
        Yson::Lookup($requestContext, "testIds")
    );
};

DEFINE ACTION $get_gauc($testId, $formulaPath, $watchCoverageThreshold) AS

-- ШАГ 1: Базовые данные с testIds
$base_data = (
    SELECT
        rid,
        $extract_test_ids(requestContext) AS test_ids,
        score,
        CAST(viewTimeSec >= $watchCoverageThreshold AS int) AS watch_label,
        CAST(ListHas(events, "actionLike") AND NOT ListHas(events, "actionUnlike") AS int) AS like_label,
        1 - CAST(ListHas(events, "actionDislike") AND NOT ListHas(events, "actionUndislike") AS int) AS dislike_label
    FROM RANGE(
        $path_to_pool_cache,
        $start_date,
        $end_date
    )
    WHERE
        formulaPath == $formulaPath
        AND typeId == $TYPE_ID
        AND userType == $USER_TYPE
        AND recommenderId == $RECOMMENDER_ID
        AND navScreen == "video_for_you"
);

-- ШАГ 2: Считаем AUC для сессий с нужным testId
$session_auc = (
    SELECT
        rid,
        ROW_NUMBER() OVER (PARTITION BY rid ORDER BY score ASC) - 1 AS rn,
        watch_label,
        like_label,
        dislike_label
    FROM $base_data
    WHERE ListHas(test_ids, $testId)
);

$session_auc_agg = (
    SELECT
        rid,
        COUNT(*) AS session_size,

        -- watch AUC (с явным приведением)
        $compute_auc(
            CAST(SUM(IF(watch_label == 1, rn + 1, 0)) AS int),
            CAST(SUM(watch_label) AS int),
            CAST(COUNT(*) AS int)
        ) AS watch_auc,

        -- like AUC
        $compute_auc(
            CAST(SUM(IF(like_label == 1, rn + 1, 0)) AS int),
            CAST(SUM(like_label) AS int),
            CAST(COUNT(*) AS int)
        ) AS like_auc,

        -- dislike AUC
        $compute_auc(
            CAST(SUM(IF(dislike_label == 1, rn + 1, 0)) AS int),
            CAST(SUM(dislike_label) AS int),
            CAST(COUNT(*) AS int)
        ) AS dislike_auc
    FROM $session_auc
    GROUP BY rid
    HAVING COUNT(*) >= 2
);

-- ШАГ 3: Финальная агрегация (с явными приведениями)
$test_gauc = (
    SELECT
        $testId AS test_id,

        -- watch GAUC
        AVG(watch_auc) AS watch_gauc_simple,
        SUM(watch_auc * CAST(session_size AS double)) / SUM(CAST(session_size AS double)) AS watch_gauc_weighted,
        COUNT(*) AS watch_sessions,
        SUM(session_size) AS watch_samples,

        -- like GAUC
        AVG(like_auc) AS like_gauc_simple,
        SUM(like_auc * CAST(session_size AS double)) / SUM(CAST(session_size AS double)) AS like_gauc_weighted,
        COUNT(*) AS like_sessions,
        SUM(session_size) AS like_samples,

        -- dislike GAUC
        AVG(dislike_auc) AS dislike_gauc_simple,
        SUM(dislike_auc * CAST(session_size AS double)) / SUM(CAST(session_size AS double)) AS dislike_gauc_weighted,
        COUNT(*) AS dislike_sessions,
        SUM(session_size) AS dislike_samples
    FROM $session_auc_agg
    WHERE watch_auc IS NOT NULL
);

-- ШАГ 4: Формируем результат (исправлено - убрал лишние запятые)
$all_metrics = (
    SELECT test_id, 'GAUCSimple_watch_coverage' AS metric, watch_gauc_simple AS value FROM $test_gauc
    UNION ALL
    SELECT test_id, 'GAUCWeighted_watch_coverage' AS metric, watch_gauc_weighted AS value FROM $test_gauc
    UNION ALL
    SELECT test_id, 'watch_sessions' AS metric, CAST(watch_sessions AS double) AS value FROM $test_gauc
    UNION ALL
    SELECT test_id, 'watch_samples' AS metric, CAST(watch_samples AS double) AS value FROM $test_gauc

    UNION ALL

    SELECT test_id, 'GAUCSimple_like' AS metric, like_gauc_simple AS value FROM $test_gauc
    UNION ALL
    SELECT test_id, 'GAUCWeighted_like' AS metric, like_gauc_weighted AS value FROM $test_gauc
    UNION ALL
    SELECT test_id, 'like_sessions' AS metric, CAST(like_sessions AS double) AS value FROM $test_gauc
    UNION ALL
    SELECT test_id, 'like_samples' AS metric, CAST(like_samples AS double) AS value FROM $test_gauc

    UNION ALL

    SELECT test_id, 'GAUCSimple_dislike' AS metric, dislike_gauc_simple AS value FROM $test_gauc
    UNION ALL
    SELECT test_id, 'GAUCWeighted_dislike' AS metric, dislike_gauc_weighted AS value FROM $test_gauc
    UNION ALL
    SELECT test_id, 'dislike_sessions' AS metric, CAST(dislike_sessions AS double) AS value FROM $test_gauc
    UNION ALL
    SELECT test_id, 'dislike_samples' AS metric, CAST(dislike_samples AS double) AS value FROM $test_gauc
);

SELECT * FROM $all_metrics
ORDER BY metric;

END DEFINE;

-- == START == --
DO $get_gauc("10029670", "fstorage:vk_video_266_1769078359_f", 30);
DO $get_gauc("10029671", "fstorage:vk_video_266_1770725758_w", 30);
DO $get_gauc("10029672", "fstorage:vk_video_266_1770737548_e", 30);

DO $get_gauc("10029670", "fstorage:vk_video_266_1769078359_f", 900);
DO $get_gauc("10029671", "fstorage:vk_video_266_1770725758_w", 900);
DO $get_gauc("10029672", "fstorage:vk_video_266_1770737548_e", 900);