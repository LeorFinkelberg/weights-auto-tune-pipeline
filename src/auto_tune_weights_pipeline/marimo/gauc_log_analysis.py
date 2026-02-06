import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from pathlib import Path
    from auto_tune_weights_pipeline.utils import LogParser

    return LogParser, Path, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## _AUX functions_
    """
    )
    return


@app.cell
def _(Path, np, pd, plt):
    def plot_auc_kde(
        aucs: np.array,
        title: str,
        text: str,
        text_x: float = 0.5,
        text_y: float = 0.5,
        save_fig: bool = False,
        figures_dir: str = "figures",
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        pd.Series(aucs).plot.kde(ax=ax)

        ax.text(text_x, text_y, text)

        ax.set_title(title)
        ax.set_xlabel("AUC")
        ax.set_xlim(0.0, 1.0)

        if save_fig:
            path_to_figures_dir = Path.cwd() / figures_dir

            if not path_to_figures_dir.exists():
                path_to_figures_dir.mkdir(exist_ok=True)

            fig.savefig(f"{path_to_figures_dir / title}.png")

        return fig

    return (plot_auc_kde,)


@app.cell
def _(Path, pd, plt):
    def plot_auc_kde_summary(
        aucs: pd.DataFrame,
        title: str,
        text: str,
        text_x: int = 0.5,
        text_y: int = 0.5,
        save_fig: bool = False,
        figures_dir: str = "figures",
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))

        aucs.plot.kde(ax=ax)

        ax.text(text_x, text_y, text)

        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("AUC")

        if save_fig:
            path_to_figures_dir = Path.cwd() / figures_dir

            if not path_to_figures_dir.exists():
                path_to_figures_dir.mkdir(exist_ok=True)

            fig.savefig(f"{path_to_figures_dir / title}.png")

        return fig

    return (plot_auc_kde_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## _Target events_
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### _action_play_
    """
    )
    return


@app.cell
def _(LogParser, np):
    aucs_2026_02_01_action_play: np.ndarray = LogParser.read_logs_to_aucs(
        "./logs/logs_for_pool_cache_2026_02_01_action_play.log",
        target_name="action_play",
    )
    return (aucs_2026_02_01_action_play,)


@app.cell
def _(aucs_2026_02_01_action_play: "np.ndarray", plot_auc_kde):
    plot_auc_kde(
        aucs_2026_02_01_action_play,
        title="pool-cache: 2026-02-01 (action play)",
        text="AUC: 0.5234\nGAUC (weighted): 0.5548\nGAUC (simple): 0.5821",
        text_x=0.05,
        text_y=3.7,
        save_fig=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### _watch_coverage_30s_
    """
    )
    return


@app.cell
def _(LogParser, np):
    aucs_2026_02_01_watch_coverage_30s: np.ndarray = LogParser.read_logs_to_aucs(
        "./logs/logs_for_pool_cache_2026_02_01_watch_coverage_30s.log",
        target_name="watch_coverage_30s",
    )
    return (aucs_2026_02_01_watch_coverage_30s,)


@app.cell
def _(LogParser, np):
    aucs_2026_02_02_watch_coverage_30s: np.ndarray = LogParser.read_logs_to_aucs(
        "./logs/logs_for_pool_cache_2026_02_02_watch_coverage_30s.log",
        target_name="watch_coverage_30s",
    )
    return (aucs_2026_02_02_watch_coverage_30s,)


@app.cell
def _(LogParser, np):
    aucs_2026_02_03_watch_coverage_30s: np.ndarray = LogParser.read_logs_to_aucs(
        "./logs/logs_for_pool_cache_2026_02_03_watch_coverage_30s.log",
        target_name="watch_coverage_30s",
    )
    return (aucs_2026_02_03_watch_coverage_30s,)


@app.cell
def _(aucs_2026_02_01_watch_coverage_30s: "np.ndarray", plot_auc_kde):
    plot_auc_kde(
        aucs_2026_02_01_watch_coverage_30s,
        title="pool-cache: 2026-02-01 (watch_coverage_30s)",
        text="AUC: 0.7105\nGAUC (weighted): 0.6115\nGAUC (simple): 0.6375",
        text_x=0.05,
        text_y=5,
        save_fig=True,
    )
    return


@app.cell
def _(aucs_2026_02_02_watch_coverage_30s: "np.ndarray", plot_auc_kde):
    plot_auc_kde(
        aucs_2026_02_02_watch_coverage_30s,
        title="pool-cache: 2026-02-02 (watch_coverage_30s)",
        text="AUC: 0.7376\nGAUC (weighted): 0.6418\nGAUC (simple): 0.6963",
        text_x=0.05,
        text_y=6,
        save_fig=True,
    )
    return


@app.cell
def _(aucs_2026_02_03_watch_coverage_30s: "np.ndarray", plot_auc_kde):
    plot_auc_kde(
        aucs_2026_02_03_watch_coverage_30s,
        title="pool-cache: 2026-02-03 (watch_coverage_30s)",
        text="AUC: 0.7339\nGAUC (weighted): 0.6368\nGAUC (simple): 0.6918",
        text_x=0.05,
        text_y=6,
        save_fig=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### _watch_coverage_60s_
    """
    )
    return


@app.cell
def _(LogParser, np):
    aucs_2026_02_01_watch_coverage_60s: np.ndarray = LogParser.read_logs_to_aucs(
        "./logs/logs_for_pool_cache_2026_02_01_watch_coverage_60s.log",
        target_name="watch_coverage_60s",
    )
    return (aucs_2026_02_01_watch_coverage_60s,)


@app.cell
def _(aucs_2026_02_01_watch_coverage_60s: "np.ndarray", plot_auc_kde):
    plot_auc_kde(
        aucs_2026_02_01_watch_coverage_60s,
        title="pool-cache: 2026-02-01 (watch_coverage_60s)",
        text="AUC: 0.7304\nGAUC (weighted): 0.6194\nGAUC (simple): 0.6429",
        text_x=0.05,
        text_y=5,
        save_fig=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### _like_
    """
    )
    return


@app.cell
def _(LogParser, np):
    aucs_2026_02_01_like: np.ndarray = LogParser.read_logs_to_aucs(
        "./logs/logs_for_pool_cache_2026_02_01_like.log", target_name="like"
    )
    return (aucs_2026_02_01_like,)


@app.cell
def _(aucs_2026_02_01_like: "np.ndarray", plot_auc_kde):
    plot_auc_kde(
        aucs_2026_02_01_like,
        title="pool-cache: 2026-02-01 (like)",
        text="",
        text_x=0.05,
        text_y=5,
        save_fig=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### _dislike_
    """
    )
    return


@app.cell
def _(LogParser, np):
    aucs_2026_02_01_dislike: np.ndarray = LogParser.read_logs_to_aucs(
        "./logs/logs_for_pool_cache_2026_02_01_dislike.log", target_name="dislike"
    )
    return (aucs_2026_02_01_dislike,)


@app.cell
def _(aucs_2026_02_01_dislike: "np.ndarray", plot_auc_kde):
    plot_auc_kde(
        aucs_2026_02_01_dislike,
        title="pool-cache: 2026-02-01 (dislike)",
        text="",
        text_x=0.05,
        text_y=5,
        save_fig=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### _Summary_
    """
    )
    return


@app.cell
def _(
    aucs_2026_02_01_action_play: "np.ndarray",
    aucs_2026_02_01_watch_coverage_30s: "np.ndarray",
    aucs_2026_02_01_watch_coverage_60s: "np.ndarray",
    aucs_2026_02_02_watch_coverage_30s: "np.ndarray",
    aucs_2026_02_03_watch_coverage_30s: "np.ndarray",
    pd,
    plot_auc_kde_summary,
):
    _min = min(
        aucs_2026_02_01_action_play.size,
        aucs_2026_02_01_watch_coverage_30s.size,
        aucs_2026_02_02_watch_coverage_30s.size,
        aucs_2026_02_03_watch_coverage_30s.size,
        aucs_2026_02_01_watch_coverage_60s.size,
    )

    plot_auc_kde_summary(
        pd.DataFrame(
            {
                "2026-02-01 (action play)": aucs_2026_02_01_action_play[:_min],
                "2026-02-01 (watch_coverage_30s)": aucs_2026_02_01_watch_coverage_30s[
                    :_min
                ],
                "2026-02-02 (watch_coverage_30s)": aucs_2026_02_02_watch_coverage_30s[
                    :_min
                ],
                "2026-02-03 (watch_coverage_30s)": aucs_2026_02_03_watch_coverage_30s[
                    :_min
                ],
                "2026-02-01 (watch_coverage_60s)": aucs_2026_02_01_watch_coverage_60s[
                    :_min
                ],
            }
        ),
        title="summary",
        text=(
            "navScreen: video_for_you,\n"
            "platform: vk_video_android,\n"
            "formulaPath: fstorage: vk_video_266_1769078359_f"
        ),
        text_x=0.05,
        text_y=3.5,
        save_fig=True,
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
