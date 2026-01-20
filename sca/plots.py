import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

def sample_trace(X, y, byte_val, seed=42):
    mask = y == byte_val
    rng = np.random.default_rng(seed=seed)
    idx = rng.integers(0, np.sum(mask))
    trace = X[mask].iloc[idx]
    return trace

def labs(ax, xlab="", ylab="", title="", **kw):
    ax.set_title(title, **kw)
    ax.set_xlabel(xlab, **kw)
    ax.set_ylabel(ylab, **kw)
    return ax

def lim(ax, l):
    ax.set_xlim(0, l-1)
    return ax

def mean_std(X):
    return X.mean(axis=0), X.std(axis=0)

def std_band(ax, m, std, sd=3):
    xaxis = np.arange(len(m))
    lower = m - sd * std
    upper = m + sd * std
    ax.fill_between(xaxis, lower, upper, color="gray", alpha=0.7, label=f"±{sd} std")
    return ax

def filtered_corr(X, t=0.2, p=0.75):
    corr = pd.DataFrame(np.corrcoef(X, rowvar=False))
    mask = np.mean(np.abs(corr) >= t, axis=0) >= p
    return corr[mask].loc[:, mask]


def ticklabsp(ax, kind, rot=0, fs=14):
    eval(f"ax.set_{kind}ticklabels(ax.get_{kind}ticklabels(),"
         f"rotation={rot}, fontsize={fs})")


def plot_trace_outliers(X, std=4):
    df, n_outliers = helpers.detect_outliers_std(X, std=std)
    
    return fig

def features_hist(X, idx, mean=False, figsize=(7, 4)):
    idx = sorted(idx)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[9, 1], wspace=0.05, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    cm = sns.color_palette("viridis", len(idx) + 1, as_cmap=True)
    norm = plt.Normalize(vmin=0, vmax=max(idx))

    for i in idx:
        sns.histplot(
            X.iloc[:, i],
            edgecolor="black",
            discrete=True,
            color=cm(norm(i)),
            stat="density",
            ax=ax,
        )
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

    lo, hi = np.percentile(X.iloc[:, idx].values, [1e-3, 100 - 1e-3])
    ax.set_xlim(lo, hi)

    if mean:
        ax.axvline(X.mean().mean(), color="black", linestyle="--")

    ax.set_title(f"Distribution of features")

    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.axis("off")

    old_legend = ax.legend([f"{i}" for i in idx], title="Feature")
    handles = old_legend.legend_handles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()

    ax_legend.legend(handles, labels, title=title, loc="center left", frameon=False)
    old_legend.remove()

    sns.despine(ax=ax)
    return fig


def dist_plots(X, figsize=(10, 4)):
    sk = stats.skew(X, axis=0, bias=False)
    kt = stats.kurtosis(X, axis=0, fisher=True, bias=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    sns.histplot(sk, ax=ax1, bins=30, stat="density", color="gray", edgecolor="black")
    ax1.set_xlabel("Skewness")
    ax1.set_ylabel("Density")
    ax1.set_title("Skewness distribution")

    sns.histplot(kt, ax=ax2, bins=30, stat="density", color="gray", edgecolor="black")
    ax2.set_xlabel("Kurtosis")
    ax2.set_ylabel("")
    ax2.set_title("Kurtosis distribution")

    # fig.tight_layout()
    sns.despine(fig)
    return fig


def class_hist(y, figsize=(14, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(y, discrete=True, edgecolor="black", ax=ax)
    ax.set_xlabel("Label")
    ax.set_title("Count of labels")
    ax.set_xlim(-0.5, 255.5)
    sns.despine(ax=ax, right=False)
    fig.tight_layout()
    return fig




def plot_feature_means(X, y, idx, figsize=(8, 5)):
    means = np.zeros((len(idx), len(np.unique(y))))

    for i, w in enumerate(np.unique((y))):
        means[:, i] = X[y == w].iloc[:, idx].mean().values

    means -= means.mean(axis=1, keepdims=True)

    df = pd.DataFrame(means.T)
    df = df.reset_index(names="label")
    df = pd.melt(df, id_vars="label", var_name="feature")
    df["feature"] = df["feature"].map({i: idx for i, idx in enumerate(idx)})

    fig, ax = plt.subplots(figsize=figsize)
    cmap = sns.color_palette("viridis", n_colors=len(np.unique(y)), as_cmap=True)

    sns.stripplot(
        data=df,
        x="feature",
        y="value",
        hue="label",
        jitter=False,
        alpha=0.7,
        palette=cmap,
        size=4,
        ax=ax,
    )

    sns.boxplot(
        data=df,
        x="feature",
        y="value",
        color="white",
        width=0.6,
        fliersize=0,
        showcaps=True,
        boxprops={"zorder": 0},
        whiskerprops={"linewidth": 0.8},
        ax=ax,
    )

    ax.grid(axis="y", linestyle="-", alpha=0.7)
    ax.legend_.remove()

    norm = plt.Normalize(vmin=min(y), vmax=max(y))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, pad=0.02, label="Label")

    ax.set_xlabel("Feature index")
    ax.set_ylabel("Centered mean")
    ax.set_title("Centered feature means by labels")
    ax.set_xticks(range(len(np.unique(idx))))
    ax.set_xticklabels(df["feature"].unique(), rotation=90)

    sns.despine(trim=True)
    fig.tight_layout()
    return fig


def plot_feature_anomalies(
    df,
    min_count=50,
    bins=[-np.inf, 4, 5, 6, 7, 10, np.inf],
    labels=["<4σ", "4–5σ", "5–6σ", "6–7σ", "7–10σ", ">10σ"],
    figsize=(9, 5),
):
    df = df.copy()
    df["Bin"] = pd.cut(df["Z-score"], bins=bins, labels=labels, right=True)

    df_feat = (
        df.groupby(["Trace", "Bin"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    df_feat["Total"] = df_feat[labels].sum(axis=1)
    df_feat = df_feat.sort_values("Total", ascending=False)
    df_feat = df_feat[df_feat["Total"] >= min_count]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(df_feat["Trace"]))
    bottom = np.zeros(len(df_feat))

    for c in labels:
        ax.bar(x, df_feat[c], bottom=bottom, label=c)
        bottom += df_feat[c].to_numpy()

    ax.set_xlabel("Trace")
    ax.set_ylabel("Count of outliers")
    ax.set_title(f"Outliers distribution (min count = {min_count})")
    ax.set_xticks(x)
    ax.set_xticklabels(df_feat["Trace"], rotation=60)

    ax.legend(title="Standard deviation")
    ax.grid(axis="y")

    sns.despine(ax=ax)
    fig.tight_layout()
    return fig




def simple_lineplots(
    x, *y, labels=None, xlabel="", ylabel="", palette="husl", figsize=(10, 5)
):
    fig, ax = plt.subplots(figsize=figsize)
    cmap = sns.color_palette(palette, n_colors=len(y))

    for i, series in enumerate(y):
        label = labels[i] if labels is not None else None
        sns.lineplot(x=x, y=series, label=label, color=cmap[i], ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.grid(True, linestyle="--", alpha=0.8)
    sns.despine(ax=ax, trim=True)

    if labels is not None:
        ax.legend()
    return fig
