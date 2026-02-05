import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from auto_tune_weights_pipeline.utils import LogParser

    return LogParser, np, pd, plt


@app.cell
def _(np, pd, plt):
    def plot_auc_kde(aucs: np.array):
        fig, ax = plt.subplots(figsize=(10, 5))
        pd.Series(aucs).plot.kde(ax=ax)

        ax.set_xlabel("AUC")
        ax.set_xlim(0.0, 1.0)

        return fig

    return (plot_auc_kde,)


@app.cell
def _(LogParser):
    aucs = LogParser.read_logs_to_aucs("./logs/app.log")
    return (aucs,)


@app.cell
def _(aucs, plot_auc_kde):
    plot_auc_kde(aucs)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
