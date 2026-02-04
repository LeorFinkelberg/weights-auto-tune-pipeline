import typing as t
import polars as pl

from dataclasses import dataclass
from auto_tune_weights_pipeline.columns import Columns
from auto_tune_weights_pipeline.events import Events


@dataclass(frozen=True)
class TargetConfig:
    name: str
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
    "action_play": TargetConfig(
        name="action_play",
        event_name=Events.ACTION_PLAY,
    ),
    "watch_coverage_30s": TargetConfig(
        name="watch_coverage_30s",
        event_name=Events.WATCH_COVERAGE_RECORD,
        view_threshold_sec=30.0,
    ),
    "watch_coverage_60s": TargetConfig(
        name="watch_coverage_60s",
        event_name=Events.WATCH_COVERAGE_RECORD,
        view_threshold_sec=60.0,
    ),
    "first_frame": TargetConfig(
        name="first_frame",
        event_name=Events.FIRST_FRAME,
    ),
}
