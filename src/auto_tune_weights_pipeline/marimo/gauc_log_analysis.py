import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from pathlib import Path
    from auto_tune_weights_pipeline.utils import LogParser
    return LogParser, Path, np, pd, plt


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


@app.cell
def _(LogParser, np):
    aucs_2026_02_01: np.ndarray = LogParser.read_logs_to_aucs("./logs/logs_for_pool_cache_2026_02_01.log")
    return (aucs_2026_02_01,)


@app.cell
def _(LogParser, np):
    aucs_2026_02_02: np.ndarray = LogParser.read_logs_to_aucs("./logs/logs_for_pool_cache_2026_02_02.log")
    return (aucs_2026_02_02,)


@app.cell
def _(LogParser, np):
    aucs_2026_02_03: np.ndarray = LogParser.read_logs_to_aucs("./logs/logs_for_pool_cache_2026_02_03.log")
    return (aucs_2026_02_03,)


@app.cell
def _(aucs_2026_02_01: "np.ndarray", plot_auc_kde):
    plot_auc_kde(
        aucs_2026_02_01,
        title="pool-cache: 2026-02-01",
        text="AUC: 0.7105\nAUC (weighted): 0.6115\nGAUC (simple): 0.6375",
        text_x=0.05,
        text_y=5,
        save_fig=True,
    )
    return


@app.cell
def _(aucs_2026_02_02: "np.ndarray", plot_auc_kde):
    plot_auc_kde(
        aucs_2026_02_02,
        title="pool-cache: 2026-02-02",
        text="AUC: 0.7376\nGAUC (weighted): 0.6418\nGAUC (simple): 0.6963",
        text_x=0.05,
        text_y=6,
        save_fig=True,
    )
    return


@app.cell
def _(aucs_2026_02_03: "np.ndarray", plot_auc_kde):
    plot_auc_kde(
        aucs_2026_02_03,
        title="pool-cache: 2026-02-03",
        text="AUC: 0.7339\nAUC (weighted): 0.6368\nGAUC (simple): 0.6918",
        text_x=0.05,
        text_y=6,
        save_fig=True,
    )
    return


@app.cell
def _(
    aucs_2026_02_01: "np.ndarray",
    aucs_2026_02_02: "np.ndarray",
    aucs_2026_02_03: "np.ndarray",
    pd,
    plot_auc_kde_summary,
):
    _min = min(aucs_2026_02_01.size, aucs_2026_02_02.size)

    plot_auc_kde_summary(
        pd.DataFrame({
            "2026-02-01": aucs_2026_02_01[:_min],
            "2026-02-02": aucs_2026_02_02[:_min],
            "2026-02-03": aucs_2026_02_03[:_min],
        }),
        title="summary",
        text=(
            "navScreen: video_for_you,\n"
            "platform: vk_video_android,\n"
            "formulaPath: fstorage: vk_video_266_1769078359_f"
        ),
        text_x=0.05,
        text_y=4.5,
        save_fig=True,
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
