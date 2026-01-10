import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="ticks")


def show_sample_trace(X, y, byte_value, ax=None, seed=42):
    mask = y == byte_value
    rng = np.random.default_rng(seed=seed)
    idx = rng.integers(0, np.sum(mask))
    trace = X[mask].iloc[idx]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))

    ax = sns.lineplot(trace, color="tab:blue", linewidth=0.8, ax=ax)
    ax.set_title(f"Sample trace (label = 0x{byte_value:02X}) ")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, len(trace) - 1)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)


def feature_histogram(X, idx, mean=False, ax=None):
    idx = sorted(idx)
    cmap = sns.color_palette("rocket", as_cmap=True)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    for i in idx:
        sns.histplot(X.iloc[:, i], color=cmap(i / len(idx)), edgecolor='k',
                     binwidth=1.0, stat="density", ax=ax)
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Density")

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, pad=0)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([idx[0], idx[-1]])

    if mean:
        ax.axvline(X.mean().mean(), color="black", linestyle="--")

    ax.set_title(f"Distribution of features")


def class_distribution(y, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax = sns.barplot(y.value_counts(), width=1, edgecolor="black", ax=ax)
    ax.set_xticks(range(0, 256, 32))
    ax.set_xlabel("Label")
    ax.set_ylabel("Counts")
    ax.set_title("Distribution of classes")


def feature_correlation(X, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    corr = np.corrcoef(X, rowvar=False)
    ax = sns.heatmap(corr, cmap="coolwarm", square=True, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
