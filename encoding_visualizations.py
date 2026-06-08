# ══════════════════════════════════════════════════════════════════════════════
#  Feature Encoding Visualizations
#  Functions: plot_hour_encodings_over_time
#             plot_holiday_encodings_over_time
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── Shared helpers ────────────────────────────────────────────────────────────

def _encoding_style(ax, label, color, ylabel=None):
    """Shared panel style for encoding plots."""
    surf = "#FFFFFF"
    ax.set_facecolor(surf)
    ax.set_ylabel(ylabel or "", fontsize=8, color="#797876", labelpad=6)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#d4d1ca")
    ax.grid(True, axis="y", alpha=0.5)
    ax.tick_params(axis="x", which="both", length=0)
    ax.text(
        0.01, 0.96, label,
        transform=ax.transAxes,
        fontsize=9, fontweight="bold", color=color,
            va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=surf,
                  edgecolor=color, alpha=0.9, linewidth=1),
    )


def _encoding_rcparams():
    """Shared rcParams for encoding plots."""
    plt.rcParams.update({
        "axes.facecolor":   "#FFFFFF",
        "axes.edgecolor":   "#d4d1ca",
        "axes.labelcolor":  "#797876",
        "xtick.color":      "#7a7974",
        "ytick.color":      "#7a7974",
        "text.color":       "#28251d",
        "grid.color":       "#ece9e4",
        "grid.linewidth":   0.5,
        "font.family":      "sans-serif",
        "figure.facecolor": "#FFFFFF",
    })


def _encoding_xaxis(axes, idx):
    """Shared x-axis formatting and day separators."""
    axes[-1].spines["bottom"].set_visible(True)
    axes[-1].spines["bottom"].set_color("#d4d1ca")
    axes[-1].tick_params(axis="x", length=3)
    axes[-1].set_xlabel("Date", fontsize=9, color="#797876")
    plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right", fontsize=8)

    day_starts = idx[idx.hour == 0]
    for ax in axes:
        for ds in day_starts:
            ax.axvline(ds, color="#d4d1ca", lw=0.6, alpha=0.6, zorder=0)


# ── Main functions ────────────────────────────────────────────────────────────

def plot_hour_encodings_over_time(
    X_onehot_full,
    X_target_full,
    X_cyclic_full,
    X_scaled_full,
    X_full,
    sample_days: int = 7,
    figsize=(16, 14),
    onehot_hours: list = None,
):
    """
    Five panels over time — all hour-of-day encodings stacked vertically,
    synchronized x-axis.

    Panels
    ------
    0 : One-Hot   — active hour as step function (from X_full["hour_of_day"])
                    or binary fills for selected hours (onehot_hours)
    1 : Target    — hour_dow_target
    2 : Cyclic    — hour_sin / hour_cos
    3 : Raw       — unscaled hour of day (int 0–23, from X_full["hour_of_day"])
    4 : Scaled    — scaled hour [0–1]  (auto-detected from X_scaled_full)

    Parameters
    ----------
    X_onehot_full : pd.DataFrame
        One-hot encoded hour columns (hour_0 … hour_23).
    X_target_full : pd.DataFrame
        Must contain "hour_dow_target".
    X_cyclic_full : pd.DataFrame
        Must contain "hour_sin" and "hour_cos".
    X_scaled_full : pd.DataFrame
        Must contain a column with "hour" in its name (e.g. "hour" or "hour_scaled").
    X_full : pd.DataFrame
        Must contain "hour" (int 0–23) and "holiday" (0/1).
    sample_days : int
        Number of days to display from the start of the index.
    figsize : tuple
        Figure size passed to plt.subplots.
    onehot_hours : list[int] | None
        None  → step function 0–23
        list  → binary fill per hour, stacked with vertical offset
    """

    teal   = "#4f98a3"
    gold   = "#e8af34"
    purple = "#a86fdf"
    orange = "#ffbd67"
    muted  = "#5a5957"
    green  = "#6daa45"
    bg     = "#FFFFFF"

    _hour_palette = [
        "#4f98a3", "#a86fdf", "#e8af34", "#fdab43", "#6daa45",
        "#5591c7", "#d163a7", "#dd6974", "#6daa45", "#e8af34",
    ]

    _encoding_rcparams()

    n   = sample_days * 24
    idx = X_full.index[:n]

    fig, axes = plt.subplots(
        5, 1, figsize=figsize, sharex=True, facecolor=bg,
        gridspec_kw={"hspace": 0.12, "top": 0.94, "bottom": 0.08},
    )
    fig.patch.set_facecolor(bg)

    # ── Panel 0: One-Hot ─────────────────────────────────────────────────────
    ax = axes[0]
    active_hour = X_full.loc[idx, "hour_of_day"].values.astype(float)

    if onehot_hours is None:
        ax.step(idx, active_hour, color=teal, lw=0.9, where="post")
        ax.set_ylim(-0.5, 24)
        ax.set_yticks([0, 6, 12, 18, 23])
        _encoding_style(ax, "One-Hot", teal, ylabel="active hour")

    else:
        offset_step  = 1.4
        total_height = offset_step * len(onehot_hours)

        for i, h in enumerate(onehot_hours):
            clr    = _hour_palette[i % len(_hour_palette)]
            signal = (active_hour == h).astype(float)
            base   = i * offset_step
            ax.fill_between(idx, base, base + signal,
                            step="post", color=clr, alpha=0.55)
            ax.step(idx, base + signal, color=clr, lw=0.8,
                    where="post", label=f"h={h:02d}")
            ax.text(idx[-1], base + 0.5, f" {h:02d}h",
                    fontsize=7, color=clr, va="center")

        ax.set_ylim(-0.3, total_height + 0.3)
        ax.set_yticks([i * offset_step + 0.5 for i in range(len(onehot_hours))])
        ax.set_yticklabels([f"{h:02d}h" for h in onehot_hours], fontsize=7)
        ax.legend(fontsize=7, loc="upper right",
                  framealpha=0.0, edgecolor="none", ncol=len(onehot_hours))
        _encoding_style(ax, f"One-Hot  (hours {onehot_hours})", teal, ylabel="active")

    # ── Panel 1: Target ───────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(idx, X_target_full.loc[idx, "hour_dow_target"], color=gold, lw=0.9)
    ax.fill_between(idx, X_target_full.loc[idx, "hour_dow_target"],
                    alpha=0.10, color=gold)
    _encoding_style(ax, "Target  (hour_dow)", gold, ylabel="target value")

    # ── Panel 2: Cyclic ───────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(idx, X_cyclic_full.loc[idx, "hour_sin"], color=purple, lw=0.9, label="sin")
    ax.plot(idx, X_cyclic_full.loc[idx, "hour_cos"], color=teal,   lw=0.9,
            label="cos", alpha=0.75, ls="--")
    ax.axhline(0, color=muted, lw=0.4, alpha=0.5)
    ax.set_ylim(-1.25, 1.25)
    ax.set_yticks([-1, 0, 1])
    ax.legend(fontsize=8, loc="upper right", framealpha=0.0, edgecolor="none")
    _encoding_style(ax, "Cyclic", purple, ylabel="sin / cos")

    # ── Panel 3: Raw hour (unscaled) ──────────────────────────────────────────
    ax = axes[3]
    ax.step(idx, active_hour, color=green, lw=0.9, where="post")
    ax.fill_between(idx, active_hour, step="post", color=green, alpha=0.10)
    ax.set_ylim(-0.5, 24)
    ax.set_yticks([0, 6, 12, 18, 23])
    _encoding_style(ax, "Raw  (hour of day)", green, ylabel="hour [0–23]")

    # ── Panel 4: Scaled ───────────────────────────────────────────────────────
    ax = axes[4]
    hour_scaled_col = next(
        (c for c in X_scaled_full.columns if "hour" in c and c != "holiday"), None
    )
    if hour_scaled_col:
        ax.plot(idx, X_scaled_full.loc[idx, hour_scaled_col], color=orange, lw=0.9)
        ax.fill_between(idx, X_scaled_full.loc[idx, hour_scaled_col],
                        alpha=0.15, color=orange)
        ax.set_yticks([0, 0.5, 1])
        _encoding_style(ax, f"Scaled  ({hour_scaled_col})", orange, ylabel="hour [0–1]")
    else:
        ax.text(0.5, 0.5, "no scaled hour column found",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color=muted)
        _encoding_style(ax, "Scaled  (hour)", orange)

    _encoding_xaxis(axes, idx)
    fig.suptitle(
        f"Hour-of-day encodings over time  ·  {sample_days}-day window",
        fontsize=12, fontweight="bold", color="#28251d",
    )
    plt.savefig("hour_encodings_over_time.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_holiday_encodings_over_time(
    X_onehot_full,
    X_target_full,
    X_cyclic_full,
    X_scaled_full,
    X_full,
    sample_days: int = 14,
    figsize=(16, 8),
):
    """
    Three panels over time — all holiday-relevant encodings stacked vertically,
    synchronized x-axis.

    Panels
    ------
    0 : One-Hot  — binary holiday flag (fill + step)
    1 : Target   — hour_holiday_target vs. hour_dow_target with delta fill;
                   holiday days highlighted via axvspan
    2 : Scaled   — holiday flag [0–1]

    Parameters
    ----------
    X_onehot_full : pd.DataFrame
        One-hot encoded hour columns (unused here, kept for API symmetry).
    X_target_full : pd.DataFrame
        Must contain "hour_dow_target" and "hour_holiday_target".
    X_cyclic_full : pd.DataFrame
        Unused, kept for API symmetry.
    X_scaled_full : pd.DataFrame
        Must contain "holiday".
    X_full : pd.DataFrame
        Must contain "holiday" (0/1).
    sample_days : int
        Number of days to display from the start of the index.
    figsize : tuple
        Figure size passed to plt.subplots.
    """

    teal           = "#4f98a3"
    gold           = "#e8af34"
    orange         = "#ffbd67"
    holiday_orange = "#fda500"
    red            = "#dd6974"
    bg             = "#FFFFFF"

    _encoding_rcparams()

    n   = sample_days * 24
    idx = X_full.index[:n]

    fig, axes = plt.subplots(
        2, 1, figsize=figsize, sharex=True, facecolor=bg,
        gridspec_kw={"hspace": 0.12, "top": 0.94, "bottom": 0.08},
    )
    fig.patch.set_facecolor(bg)

    # ── Panel 0: One-Hot holiday flag ─────────────────────────────────────────
    ax = axes[0]
    holiday_flag = X_full.loc[idx, "holiday"].values.astype(float)
    ax.fill_between(idx, holiday_flag, step="post", color=holiday_orange, alpha=0.45)
    ax.step(idx, holiday_flag, color=holiday_orange, lw=1.2, where="post")
    ax.set_ylim(-0.15, 1.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0", "1"], fontsize=8)
    _encoding_style(ax, "One-Hot  (holiday flag)", holiday_orange, ylabel="holiday")

    # ── Panel 1: Target — holiday vs. dow ─────────────────────────────────────
    ax = axes[1]
    dow = X_target_full.loc[idx, "hour_dow_target"]
    hol = X_target_full.loc[idx, "hour_holiday_target"]

    ax.plot(idx, dow, color=gold, lw=0.9, label="hour_dow_target",     alpha=0.9)
    ax.plot(idx, hol, color=red,  lw=0.9, label="hour_holiday_target", ls="--", alpha=0.85)
    ax.fill_between(idx, dow, hol, where=(hol > dow),
                    alpha=0.18, color=red,  label="holiday > dow")
    ax.fill_between(idx, dow, hol, where=(hol < dow),
                    alpha=0.18, color=gold, label="holiday < dow")

    holiday_idx = idx[X_full.loc[idx, "holiday"] == 1]
    for hd in holiday_idx[::24]:
        ax.axvspan(hd, hd + pd.Timedelta(hours=23),
                   color=holiday_orange, alpha=0.07, zorder=0)

    ax.legend(fontsize=7, loc="upper right", framealpha=0.0, edgecolor="none", ncol=2)
    _encoding_style(ax, "Target  (holiday vs. dow)", gold, ylabel="target value")

    return fig