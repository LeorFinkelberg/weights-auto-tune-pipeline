from enum import StrEnum


class Columns(StrEnum):
    RID_COL_NAME = "rid"
    SCORE_COL_NAME = "score"
    EVENTS_COL_NAME = "events"
    VIEW_TIME_SEC_COL_NAME = "viewTimeSec"
    WATCH_COVERAGE_RECORD_COL_NAME = "watchCoverageRecord"
    FIRST_FRAME_COL_NAME = "firstFrame"
