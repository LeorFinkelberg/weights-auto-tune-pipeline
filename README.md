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
#### Расчет GAUC

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

#### Обучение ML-модели
```python
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from auto_tune_weights_pipeline.maxtrix_builder import AutoMatrixBuilder

matrix_builder = AutoMatrixBuilder(
    path_to_pool_cache_with_features="data/pool_cache_with_features_2026_02_01.jsonl"
)
_X, _y = matrix_builder.get_Xy_matrix(target_label="actionPlay")
X = _X.to_numpy()
y = _y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logger.debug("Training model ...")
clf = LogisticRegressionCV()
clf.fit(X_train, y_train)
logger.debug(roc_auc_score(y_test, clf.predict(X_test)))
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