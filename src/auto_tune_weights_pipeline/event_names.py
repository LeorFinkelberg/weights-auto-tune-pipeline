from enum import StrEnum


class EventNames(StrEnum):
    ACTION_PLAY = "actionPlay"
    LIKE = "actionLike"
    DISLIKE = "actionDislike"
    WATCH_COVERAGE_RECORD = "watchCoverageRecord"
    FIRST_FRAME = "firstFrame"
