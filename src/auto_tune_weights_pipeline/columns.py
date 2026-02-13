from enum import StrEnum


class Columns(StrEnum):
    RID_COL_NAME = "rid"
    SCORE_COL_NAME = "score"
    CATBOOST_SCORE_COL_NAME = "catboost_score"
    EVENTS_COL_NAME = "events"
    VIEW_TIME_SEC_COL_NAME = "viewTimeSec"
    NAV_SCREEN_COL_NAME = "navScreen"
    PLATFORM_COL_NAME = "platform"
    FORMULA_PATH_COL_NAME = "formulaPath"
    FEATURES_COL_NAME = "features"
