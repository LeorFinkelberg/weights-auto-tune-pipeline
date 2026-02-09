use jupiter;

-- === GET SUBSAMPLE FROM POOL-CACHE ===
pragma yt.DefaultOperationWeight = "10.0";
pragma yt.MaxRowWeight = "128M";
pragma yt.StaticPool = "ucp-vkvideo-pool-cache";

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