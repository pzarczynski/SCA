import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts import util


def labs(ax, xlab="", ylab="", title="", **kw):
    ax.set_title(title, **kw, size=16)
    ax.set_xlabel(xlab, **kw, size=14)
    ax.set_ylabel(ylab, **kw, size=14)
    return ax


def savetight(fig, name, **kw):
    fig.savefig(f"figures/{name}.png", bbox_inches='tight', pad_inches=0.02, dpi=300)


def std_band(ax, m, std, sd=3, x=None, color='lightgray', alpha=0.3):
    xaxis = np.arange(len(m)) if x is None else x
    lower = m - sd * std
    upper = m + sd * std
    ax.fill_between(xaxis, lower, upper, color=color, alpha=alpha)
    return ax


def center0(axarr):
    y_min, y_max = zip(*(ax.get_ylim() for ax in axarr.ravel()))
    y_abs_max = max(abs(min(y_min)), abs(max(y_max)))
    for ax in axarr.ravel():
        ax.set_ylim(-y_abs_max, y_abs_max)


def rot_labels(axarr, rot=45, kind='x'):
    axarr = np.array(axarr)
    for ax in axarr.ravel():
        ticklabsp(ax, kind, rot=rot)


def ticklabsp(ax, kind, rot=0):
    eval(f"ax.set_{kind}ticklabels(ax.get_{kind}ticklabels(),"
         f"rotation={rot})")


def plot_mean_std(X, figsize=(8, 5), ax=None, color='gray', alpha=0.7):
    if ax is None: _, ax = plt.subplots(figsize=figsize)
    m, sd = util.mean_std(X)
    ax.plot(range(len(m)), m, linewidth=1, color='black')
    ax.set_xlim (0, len(m))
    std_band(ax, m, sd, color=color, alpha=alpha)
    ax.grid(True, linestyle="--", alpha=0.8)
    return ax.figure, ax


def plot_feature_means(X, y, figsize=(8, 5), box=True):
    means = np.array([X[y == w].mean(axis=0) for w in np.unique(y)]).T
    means -= means.mean(axis=1, keepdims=True)

    df = pd.DataFrame(means.T)
    df = df.reset_index(names="label")
    df = pd.melt(df, id_vars="label", var_name="feature")

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

    if box:
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

    ax.set_xlabel("Time samples")
    ax.set_ylabel("Centered mean")
    ax.set_title("Centered feature means by labels")
    ax.set_xticks([])
    # ax.set_xticklabels(df["feature"].unique(), rotation=90)

    sns.despine(trim=True)
    fig.tight_layout()
    return fig


def plot_scores(scores, n=10):
    df = pd.DataFrame(scores, columns=["Score"]).reset_index(names="Key")
    df["Key"] = df["Key"].astype(str) + '/' + [str(x) for x in h.hw(df["Key"])]
    df = df.sort_values("Score", ascending=False)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(5, 6))

    ax1.bar(df["Key"].iloc[:n], df["Score"].iloc[:n], width=0.9)
    ax1.set_ylabel("Perceived information")
    ticklabsp(ax1, 'x', rot=-60)
    ax1.axhline(0.0, color='k')

    ax2.bar(df["Key"].iloc[-n:], df["Score"].iloc[-n:], width=0.9)
    ax2.set_ylabel("Perceived information")
    ticklabsp(ax2, 'x', rot=-60)
    ax2.axhline(0.0, color='k')

    ax2.set_xlabel("Time samples/HW")
    fig.tight_layout(h_pad=1.0)
    return fig


def plot_score(means, stds=None, frs=None, n=500, name="Guessing entropy"):
    fig, ax = plt.subplots(figsize=(11, 4))
    m = means[:n]

    ax.plot(range(len(m)), m, linewidth=1, label="mean")
    labs(ax, xlab="Number of traces", ylab=name)
    ax.set_xlim (0, len(m))

    if stds is not None:
        std_band(ax, m, stds[:n], sd=1).grid(True, linestyle="--", alpha=0.8)

    if frs is not None:
        mfrs = np.mean(frs)

        if mfrs < n:
            if not isinstance(frs, int):
                mfrs, sdfts = np.mean(frs), np.std(frs)
                plt.fill_betweenx(ax.get_ylim(), mfrs-sdfts, mfrs+sdfts,
                                color='red', alpha=0.2, label='Std. deviation band')

            ax.axvline(x=mfrs, color='red', linestyle='--', label='Avg. traces needed')

    ax.grid(True, linestyle="--", alpha=0.8)

    ax.legend(loc='upper right')
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig, ax


def plot_cv_results(
    df, title="CV scores", ax=None, xlab='Fold',
    ylab='Perceived Information', plot_prof=True
):
    x = range(len(df))

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.errorbar(x, df["atk_score_mean"],
                yerr=df["atk_score_std"],
                fmt="s-", capsize=3, label="Attack",)

    if plot_prof:
        ax.errorbar(x, df["prof_score_mean"],
                    yerr=df["prof_score_std"],
                    fmt="o-", capsize=3, label="Profiling",)

    labs(ax, xlab=xlab, ylab=ylab,
         title=f"{title}\n"
               f"Mean score={df['atk_score_mean'].mean():.4f} "
               f"(±{df['atk_score_mean'].std():.4f})")

    ax.axhline(0.0, color='black', linestyle="--", alpha=0.8, label='Random guessing')

    ax.grid(True, linestyle="--", alpha=0.8)
    ax.legend()
    plt.tight_layout()
    return ax.figure, ax


def plot_gs_results(
    df, title, ax=None, key='n', xlab='POI',
    ylab='Perceived Information', plot_prof=True, loc=None
):
    if isinstance(df, str):
        df = pd.read_csv(f'data/results/{df}.csv')
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    xaxis = np.arange(df.shape[0])

    ax.errorbar(
        xaxis, df["atk_score_mean"],
        yerr=df["atk_score_std"], fmt='o-', loc=None,
        capsize=3, label="Attack", color='darkblue',
    )

    ax.grid(True, linestyle="--", alpha=0.8)
    ax.set_xticks(xaxis)

    ticklabels = df[key].astype(str)
    if 'expl' in df.columns:
        ticklabels += ' (' + (df["expl"]*100).astype(int).astype(str) + '%)'

    ax.set_xticklabels(ticklabels)
    ax.tick_params(axis='y', labelcolor='darkblue')

    i = df['atk_score_mean'].argmax()
    best_val = df['atk_score_mean'].iloc[i]
    labs(ax, xlab=xlab, ylab=ylab, title=f"{title}\n"
            f"Best: {best_val:.4f} "
            f"(±{df['atk_score_std'].iloc[i]:.4f}) ")
    ax.scatter(i, best_val, marker='^', s=100, color='red', zorder=3, label='Best score')

    ax.axhline(0.0, color='black', linestyle="--", alpha=0.8, label='Random guessing')

    lines, labels = ax.get_legend_handles_labels()

    if plot_prof:
        twinx = ax.twinx()
        twinx.errorbar(
            xaxis + 0.1, df["prof_score_mean"],
            yerr=df["prof_score_std"], fmt='s-',
            capsize=3, label="Profiling", color='brown',
        )
        lines_tw, labels_tw = twinx.get_legend_handles_labels()
        lines, labels = (lines + lines_tw), (labels + labels_tw)
        twinx.tick_params(axis='y', labelcolor='brown')

    ax.legend(lines, labels, loc=loc)
    ax.figure.tight_layout()

    if isinstance(df, str):
        ax.figure.savefig(f"figures/{df}.png", bbox_inches='tight',
                    pad_inches=0.02, dpi=300)
    return ax.figure, ax
