## _Scenarios for working with the library_

NB! Before each commit, a `pre-commit` is run with verification. However, if you want to commit changes without `pre-commit`, then add the `--no-verify` flag.
```bash
$ git commit -v --no-verify
```

### _Tune weights for target events_
```bash
# Download pool_caches
$ yt --proxy jupiter.yt.vk.team read "//home/.../pool_cache_features_2026_02_01_train" \
    --format "<stringify_nan_and_infinity=%true>json" > ./data/pool_cache_with_features_2026_02_01_train.jsonl
$ yt --proxy jupiter.yt.vk.team read "//home/.../pool_cache_features_2026_02_01_val" \
    --format "<stringify_nan_and_infinity=%true>json" > ./data/pool_cache_with_features_2026_02_01_val.jsonl
# Run tunner
$ uv run cli.py --help
$ uv run cli.py \
    --path-to-pool-cache-train ./data/pool_cache_with_features_2026_02_01_train.jsonl
    --path-to-pool-cache-val ./data/pool_cache_with_features_2026_02_02_val.jsonl
    --loss-function PairLogitPairwise \
    --n-trials 10
    --timeout 600
# Or just use bash script
$ chmod +x run_tune_pipeline.sh
$ ./run_tune_pipeline.sh
# Optuna dashboards
$ uv run optuna-dashboard sqlite:///tune_target_events_weights.db
# Listening on http://127.0.0.1:8080/
# Hit Ctrl-C to quit.
```
### _MARIMO / JupterHub etc._
```bash
$ git clone <repo>
$ cd <repo>
$ uv sync
$ uv run marimo run
```
#### _GAUC Compute_

In `marimo` session ...
```python
import typing as t
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
    platforms=("vk_video_android",),
    calculate_regular_auc=True,
)

summary = gauc_metric.get_summary(results)
print(summary)
```

#### _ML-model training_

```python
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from auto_tune_weights_pipeline.events import Events
from auto_tune_weights_pipeline.maxtrix_builder import AutoMatrixBuilder

X_train, y_train, X_test, y_test = AutoMatrixBuilder.train_test_split(
    path_to_pool_cache_train="./data/pool_cache_with_features_2026_02_01_train.jsonl",
    path_to_pool_cache_test="./data/pool_cache_with_features_2026_02_02_test.jsonl",
    target_label=Events.ACTION_PLAY,
    max_features=213,
)

pipeline = Pipeline(
    [
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        (
            "logRegCV",
            LogisticRegression(
                C=10.0,
                l1_ratio=0.0,
                max_iter=5_000,
            ),
        ),
    ]
)

print("Training model ...")
pipeline.fit(X_train, y_train)
print("ROC AUC: {:.5f}".format(roc_auc_score(y_test, pipeline.predict(X_test))))
```

### _Terminal_
```bash
$ git clone <repo>
$ cd <repo>
$ uv sync
```
### _Tests_
Run test
```bash
$ pytest
$ pytest -vv -m auc  # only for AUC computing test
```