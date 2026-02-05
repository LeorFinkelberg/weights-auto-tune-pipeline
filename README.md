## _Сценарии работы с библиотекой_

NB! Перед каждым коммитом запускается `pre-commit` с проверкой. Однако, если требуется зафиксировать изменения без `pre-commit`, то следует добавить флаг `--no-verify`
```bash
$ git commit -v --no-verify
```
### _MARIMO / JupterHub etc._
```bash
$ git clone <repo>
$ cd <repo>
$ uv sync
$ uv run marimo run
```
In `marimo` session ...
```python
from auto_tune_weights_pipeline.columns import Columns
from auto_tune_weights_pipeline.metrics.gauc import GAUC
from auto_tune_weights_pipeline.target_config import TargetConfig
from auto_tune_weights_pipeline.events import Events

target_config: t.Final[dict] = {
    "watch_coverage_30s": TargetConfig(
        name="watch_coverage_30s",
        event_name=Events.WATCH_COVERAGE_RECORD,
        view_threshold_sec=30.0,
    )
}

gauc_metric = GAUC(path_to_pool_cache="./data/pool_cache_2026_01_31.jsonl")
results = gauc_metric.calculate_metric(
    target_configs=target_config,
    session_col_name=Columns.RID_COL_NAME,
    nav_screen="video_for_you",
    platform="vk_video_android",
    calculate_regular_auc=True,
)

summary = gauc_metric.get_summary(results)
print(summary)
```
### _Terminal_
```bash
$ git clone <repo>
$ cd <repo>
$ uv sync
```
### _Tests_
Для запуска тестов выполнить
```bash
$ nox -s test
```