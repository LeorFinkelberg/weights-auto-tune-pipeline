import typing as t
import polars as pl

from enum import StrEnum, auto
from dataclasses import dataclass
from auto_tune_weights_pipeline.columns import Columns
from auto_tune_weights_pipeline.event_names import EventNames


class TargetNames(StrEnum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values) -> str:
        return name.lower()

    ACTION_PLAY = auto()
    WATCH_COVERAGE_30S = auto()
    WATCH_COVERAGE_60S = auto()
    FIRST_FRAME = auto()


@dataclass(frozen=True)
class TargetConfig:
    target_name: str
    event_name: str
    view_threshold_sec: t.Optional[float] = None
    condition: t.Optional[t.Callable[[pl.DataFrame], pl.Expr]] = None

    def create_label_expr(self, pool_cache: t.Optional[pl.DataFrame] = None) -> pl.Expr:
        base_expr = pl.col(Columns.EVENTS_COL_NAME).list.contains(self.event_name)

        if self.view_threshold_sec is not None:
            base_expr = base_expr & (
                pl.col(Columns.VIEW_TIME_SEC_COL_NAME) >= self.view_threshold_sec
            )

        if self.condition is not None and pool_cache is not None:
            base_expr = base_expr & self.condition(pool_cache)

        return pl.when(base_expr).then(1).otherwise(0)


DEFAULT_TARGETS_CONFIG: t.Final[dict] = {
    TargetNames.ACTION_PLAY: TargetConfig(
        target_name=TargetNames.ACTION_PLAY,
        event_name=EventNames.ACTION_PLAY,
    ),
    TargetNames.WATCH_COVERAGE_30S: TargetConfig(
        target_name=TargetNames.WATCH_COVERAGE_30S,
        event_name=EventNames.WATCH_COVERAGE_RECORD,
        view_threshold_sec=30.0,
    ),
    TargetNames.WATCH_COVERAGE_60S: TargetConfig(
        target_name=TargetNames.WATCH_COVERAGE_60S,
        event_name=EventNames.WATCH_COVERAGE_RECORD,
        view_threshold_sec=60.0,
    ),
    TargetNames.FIRST_FRAME: TargetConfig(
        target_name=TargetNames.FIRST_FRAME,
        event_name=EventNames.FIRST_FRAME,
    ),
}
