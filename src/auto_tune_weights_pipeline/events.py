from enum import StrEnum


class Events(StrEnum):
    ACTION_PLAY = "actionPlay"
    WATCH_COVERAGE_RECORD = "watchCoverageRecord"
    FIRST_FRAME = "firstFrame"
