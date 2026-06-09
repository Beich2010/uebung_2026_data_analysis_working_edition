#
#
# HELPER METHODS BLOCK Do not change this block
#
#



















#
#
#
#
#
#
#




from ipywidgets import interact, IntSlider, fixed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sktime.transformations.series.date import DateTimeFeatures
from sktime.transformations.series.holiday import HolidayFeatures
from holidays import country_holidays
from chronos import Chronos2Pipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def plot_forecasts_for_day(
    models_dict,
    day,
    y=None,
    X_full=None,
    X_target_full=None,
    X_cyclic_full=None,
    X_scaled_full=None,
    context_hours=48,
):
    """
    Plot forecasts for a given day with optional context window, y overlay,
    hour encodings, and holiday encodings.

    Panels (top → bottom)
    ---------------------
    0        : Forecast + context + y overlay (always shown)
    1        : y (true load series over context+forecast) — if y provided
    --- Hour encodings (if X_full provided) ---
    n+0      : Raw hour_of_day [0–23]
    n+1      : Target  hour_dow_target          — if X_target_full provided
    n+2      : Cyclic  hour_sin / hour_cos       — if X_cyclic_full provided
    n+3      : Scaled  hour [0–1]               — if X_scaled_full provided
    --- Holiday encodings (if X_full + X_target_full provided) ---
    n+4      : One-Hot  holiday flag (binary)
    n+5      : Target   hour_holiday_target vs. hour_dow_target

    Args:
        models_dict (dict):  model name → DataFrame with 'y_pred', 'y_test'.
        day (int):           Day index (1-based).
        y (pd.Series|None):  Full ground-truth series (DatetimeIndex) for
                             context window ground truth + optional overlay.
        X_full (pd.DataFrame|None):        Must contain "hour_of_day", "holiday".
        X_target_full (pd.DataFrame|None): Must contain "hour_dow_target",
                                           "hour_holiday_target".
        X_cyclic_full (pd.DataFrame|None): Must contain "hour_sin", "hour_cos".
        X_scaled_full (pd.DataFrame|None): Must contain a column with "hour".
        context_hours (int): Hours of context before the forecast window.
    """

    # ── Colors ────────────────────────────────────────────────────────────────
    teal           = "#4f98a3"
    gold           = "#e8af34"
    purple         = "#a86fdf"
    orange         = "#ffbd67"
    holiday_orange = "#fda500"
    red            = "#dd6974"
    green          = "#6daa45"
    muted          = "#5a5957"
    bg             = "#FFFFFF"

    # ── Forecast data ─────────────────────────────────────────────────────────
    first_df   = next(iter(models_dict.values()))
    row        = first_df.iloc[day - 1]
    y_true     = row["y_test"]
    x          = y_true.index          # 24 forecast timestamps
    y_true_arr = np.array(y_true)

    # ── Context index ─────────────────────────────────────────────────────────
    show_context   = context_hours > 0
    show_encodings = X_full is not None
    show_y_panel   = y is not None

    ctx_start = x[0] - pd.Timedelta(hours=context_hours)

    if show_context and y is not None:
        ctx_y   = y[(y.index >= ctx_start) & (y.index < x[0])]
        ctx_idx = ctx_y.index
    elif show_context and X_full is not None:
        ctx_idx = X_full.index[(X_full.index >= ctx_start) & (X_full.index < x[0])]
        ctx_y   = None
    else:
        ctx_idx = pd.DatetimeIndex([])
        ctx_y   = None

    enc_idx = ctx_idx.append(x) if len(ctx_idx) else x

    # ── Panel count ───────────────────────────────────────────────────────────
    show_hour_raw    = show_encodings
    show_hour_target = show_encodings and X_target_full is not None
    show_hour_cyclic = show_encodings and X_cyclic_full is not None
    show_hour_scaled = show_encodings and X_scaled_full is not None
    show_hol_onehot  = show_encodings and "holiday" in X_full.columns
    show_hol_target  = show_encodings and X_target_full is not None and \
                       "hour_holiday_target" in X_target_full.columns

    n_enc = sum([show_hour_raw, show_hour_target, show_hour_cyclic,
                 show_hour_scaled, show_hol_onehot, show_hol_target])
    n_rows  = 1 + (1 if show_y_panel else 0) + n_enc
    heights = [3] + ([1] if show_y_panel else []) + [1] * n_enc

    _encoding_rcparams()

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(14, 3 + 1.6 * max(n_enc + (1 if show_y_panel else 0), 1)),
        sharex=True,
        gridspec_kw={
            "height_ratios": heights,
            "hspace":        0.12,
            "top":           0.94,
            "bottom":        0.08,
        },
    )
    if n_rows == 1:
        axes = [axes]
    fig.patch.set_facecolor(bg)

    ax_fc    = axes[0]
    ax_iter  = iter(axes[1:])

    # ── Holiday axvspan across all panels ─────────────────────────────────────
    if show_hol_onehot:
        is_hol_fc = X_full.loc[x, "holiday"].values.astype(bool)
        if is_hol_fc.any():
            for ax in axes:
                ax.axvspan(x[0], x[-1],
                           color=holiday_orange, alpha=0.06, zorder=0)
        if show_context and len(ctx_idx):
            ctx_hol = X_full.loc[ctx_idx, "holiday"].values.astype(bool)
            for i, ts in enumerate(ctx_idx):
                if ctx_hol[i]:
                    for ax in axes:
                        ax.axvspan(ts, ts + pd.Timedelta(hours=1),
                                   color=holiday_orange, alpha=0.06, zorder=0)

    # ── Panel 0: Forecast ─────────────────────────────────────────────────────
    # context ground truth
    if show_context and ctx_y is not None and len(ctx_y):
        ax_fc.plot(ctx_y.index, np.array(ctx_y),
                   color=muted, lw=1.0, alpha=0.5, label="context (true)")
    elif show_context and ctx_y is None and y is not None:
        pass  # already handled above

    if show_context and len(ctx_idx):
        ax_fc.axvline(x[0], color="#d4d1ca", lw=1.0, ls="--", alpha=0.8)

    ax_fc.plot(x, y_true_arr, label="y (True)", lw=2.0, color="#28251d")
    for lbl, df in models_dict.items():
        yp = df.iloc[day - 1]["y_pred"]
        if hasattr(yp, "index"):
            yp = np.array(yp)
        ax_fc.plot(x, yp, label=lbl, ls="--", lw=1.1)

    # holiday label
    if show_hol_onehot and X_full.loc[x, "holiday"].values.astype(bool).any():
        ax_fc.set_title(
            f"Forecast Comparison – {x[0].date()} is Holiday",
            fontsize=11, fontweight="bold", color="#28251d", loc="left",
        )
    else:
        ax_fc.set_title(
            f"Forecast Comparison – {x[0].date()}",
            fontsize=11, fontweight="bold", color="#28251d", loc="left",
        )

    ax_fc.set_ylabel("Value", fontsize=8, color="#797876")
    ax_fc.grid(True, axis="y", alpha=0.4)
    ax_fc.spines[["top", "right"]].set_visible(False)
    ax_fc.legend(fontsize=7, framealpha=0.0, ncol=2)

    # ── Panel y (full series) ─────────────────────────────────────────────────
    if show_y_panel:
        ax = next(ax_iter)
        y_enc = y[y.index.isin(enc_idx)] if y is not None else None
        if y_enc is not None and len(y_enc):
            ax.plot(y_enc.index, np.array(y_enc),
                    color=muted, lw=0.9, alpha=0.85)
            ax.fill_between(y_enc.index, np.array(y_enc),
                            alpha=0.10, color=muted)
        if show_context and len(ctx_idx):
            ax.axvline(x[0], color="#d4d1ca", lw=1.0, ls="--", alpha=0.8)
        _encoding_style(ax, "y  (true load)", muted, ylabel="load")

    # ── Hour encoding panels ──────────────────────────────────────────────────
    def _vline(ax):
        if show_context and len(ctx_idx):
            ax.axvline(x[0], color="#d4d1ca", lw=1.0, ls="--", alpha=0.8)

    if show_hour_raw:
        ax = next(ax_iter)
        hv = X_full.loc[enc_idx, "hour_of_day"].values.astype(float)
        ax.step(enc_idx, hv, color=green, lw=0.9, where="post")
        ax.fill_between(enc_idx, hv, step="post", color=green, alpha=0.12)
        ax.set_ylim(-0.5, 24)
        ax.set_yticks([0, 6, 12, 18, 23])
        _vline(ax)
        _encoding_style(ax, "Raw  (hour_of_day)", green, ylabel="hour [0–23]")

    if show_hour_target:
        ax = next(ax_iter)
        ax.plot(enc_idx, X_target_full.loc[enc_idx, "hour_dow_target"],
                color=gold, lw=0.9)
        ax.fill_between(enc_idx, X_target_full.loc[enc_idx, "hour_dow_target"],
                        alpha=0.10, color=gold)
        _vline(ax)
        _encoding_style(ax, "Target  (hour_dow)", gold, ylabel="target")

    if show_hour_cyclic:
        ax = next(ax_iter)
        ax.plot(enc_idx, X_cyclic_full.loc[enc_idx, "hour_sin"],
                color=purple, lw=0.9, label="sin")
        ax.plot(enc_idx, X_cyclic_full.loc[enc_idx, "hour_cos"],
                color=teal, lw=0.9, label="cos", ls="--", alpha=0.75)
        ax.axhline(0, color=muted, lw=0.4, alpha=0.5)
        ax.set_ylim(-1.25, 1.25)
        ax.set_yticks([-1, 0, 1])
        ax.legend(fontsize=7, loc="upper right", framealpha=0.0, edgecolor="none")
        _vline(ax)
        _encoding_style(ax, "Cyclic  (hour)", purple, ylabel="sin/cos")

    if show_hour_scaled:
        ax = next(ax_iter)
        col = next((c for c in X_scaled_full.columns
                    if "hour" in c and c != "holiday"), None)
        if col:
            ax.plot(enc_idx, X_scaled_full.loc[enc_idx, col],
                    color=orange, lw=0.9)
            ax.fill_between(enc_idx, X_scaled_full.loc[enc_idx, col],
                            alpha=0.15, color=orange)
            ax.set_yticks([0, 0.5, 1])
            _vline(ax)
            _encoding_style(ax, f"Scaled  ({col})", orange, ylabel="[0–1]")

    # ── Holiday encoding panels ───────────────────────────────────────────────
    if show_hol_onehot:
        ax = next(ax_iter)
        hf = X_full.loc[enc_idx, "holiday"].values.astype(float)
        ax.fill_between(enc_idx, hf, step="post", color=holiday_orange, alpha=0.45)
        ax.step(enc_idx, hf, color=holiday_orange, lw=1.2, where="post")
        ax.set_ylim(-0.15, 1.4)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["0", "1"], fontsize=8)
        _vline(ax)
        _encoding_style(ax, "One-Hot  (holiday)", holiday_orange, ylabel="holiday")

    if show_hol_target:
        ax = next(ax_iter)
        dow = X_target_full.loc[enc_idx, "hour_dow_target"]
        hol = X_target_full.loc[enc_idx, "hour_holiday_target"]
        ax.plot(enc_idx, dow, color=gold, lw=0.9,
                label="hour_dow_target", alpha=0.9)
        ax.plot(enc_idx, hol, color=red,  lw=0.9,
                label="hour_holiday_target", ls="--", alpha=0.85)
        ax.fill_between(enc_idx, dow, hol, where=(hol > dow),
                        alpha=0.18, color=red,  label="holiday > dow")
        ax.fill_between(enc_idx, dow, hol, where=(hol < dow),
                        alpha=0.18, color=gold, label="holiday < dow")
        hol_idx = enc_idx[X_full.loc[enc_idx, "holiday"] == 1]
        for hd in hol_idx[::24]:
            ax.axvspan(hd, hd + pd.Timedelta(hours=23),
                       color=holiday_orange, alpha=0.07, zorder=0)
        ax.legend(fontsize=7, loc="upper right",
                  framealpha=0.0, edgecolor="none", ncol=2)
        _vline(ax)
        _encoding_style(ax, "Target  (holiday vs. dow)", gold, ylabel="target")

    # ── x-axis ────────────────────────────────────────────────────────────────
    axes[-1].spines["bottom"].set_visible(True)
    axes[-1].spines["bottom"].set_color("#d4d1ca")
    axes[-1].tick_params(axis="x", length=3)
    axes[-1].set_xlabel("Timestamp", fontsize=9, color="#797876")
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    plt.show()


def interactive_forecast_plot(
    models_dict,
    y=None,
    X_full=None,
    X_target_full=None,
    X_cyclic_full=None,
    X_scaled_full=None,
    context_hours=48,
    max_context_hours=24 * 7 * 4 * 4 * 2,
):
    """
    Interactive slider to browse forecast plots per day.

    Usage (minimal):
        interactive_forecast_plot(models_dict)

    Usage (full):
        interactive_forecast_plot(
            models_dict,
            y=y_series,             # pd.Series with full DatetimeIndex
            X_full=X_full,          # must contain "hour_of_day", "holiday"
            X_target_full=X_target_full,
            X_cyclic_full=X_cyclic_full,
            X_scaled_full=X_scaled_full,
            context_hours=48,
        )
    """
    num_days = len(next(iter(models_dict.values())))

    interact(
        plot_forecasts_for_day,
        models_dict=fixed(models_dict),
        day=IntSlider(min=1, max=num_days, step=1, value=1, description="Day"),
        y=fixed(y),
        X_full=fixed(X_full),
        X_target_full=fixed(X_target_full),
        X_cyclic_full=fixed(X_cyclic_full),
        X_scaled_full=fixed(X_scaled_full),
        context_hours=IntSlider(min=0, max=max_context_hours, step=24, value=context_hours,
                                description="Context h"),
    )
#
#
#

def plot_full_forecast_series(models_dict):
    """
    Plot the full time series of y_test and all y_pred across all days, using native timestamps.

    Assumes y_test and y_pred in each row are pandas Series (with DateTimeIndex).
    """
    # Get full y_true time series
    first_df = next(iter(models_dict.values()))
    y_true_series = pd.concat(first_df["y_test"].values)

    plt.figure(figsize=(14, 5))
    plt.plot(y_true_series.index, y_true_series.values, label="y (True)", linewidth=2)

    # Plot all model predictions
    for label, df in models_dict.items():
        y_pred_series = pd.concat(df["y_pred"].values)
        plt.plot(y_pred_series.index, y_pred_series.values, label=label, linestyle="--")

    plt.title("Full Forecast Series Over Time")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


def plot_full_forecast_series_prob(models_dict):
    """
    Plot the full time series of y_test and all y_pred across all days, using native timestamps.

    Assumes y_test and y_pred in each row are pandas Series (with DateTimeIndex).
    """
    # Get full y_true time series
    first_df = next(iter(models_dict.values()))
    y_true_series = pd.concat(first_df["y_test"].values)

    plt.figure(figsize=(14, 5))
    plt.plot(y_true_series.index, y_true_series.values, label="y (True)", linewidth=2)

    # Plot all model predictions
    for label, df in models_dict.items():
        y_pred_df = pd.concat(df["y_pred_quantiles"].values)
        # remove the multiindex from the DataFrame
        y_pred_df.columns = y_pred_df.columns.droplevel(0)  #
        #iterate over the quantiles
        for quantile in y_pred_df.columns:
            y_pred_series = y_pred_df[quantile]
            plt.plot(y_pred_series.index, y_pred_series.values, label=f"{label} - {quantile}", linestyle="--")
        

    plt.title("Full Forecast Series Over Time")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()



#
# Summary Metrics Function
#

def summarize_metrics(models_dict, prefix="test_"):
    """
    Generate an evaluation table from precomputed test metrics in the DataFrames.

    Args:
        models_dict (dict): Dictionary with model names as keys and DataFrames as values.
                            Each DataFrame should contain columns like 'test_MAE', 'test_MSE', etc.
        prefix (str): Prefix of metric columns (default is 'test_').

    Returns:
        pd.DataFrame: Evaluation table with models as rows and metrics as columns (mean per metric).
    """
    results = {}

    for label, df in models_dict.items():
        # Select only columns with the given prefix
        metric_cols = [col for col in df.columns if col.startswith(prefix)]
        
        # Remove the prefix in column names
        renamed_metrics = {col: col[len(prefix):] for col in metric_cols}
        
        # Compute mean for each metric
        means = df[metric_cols].mean().rename(index=renamed_metrics)
        
        results[label] = means

    # sort by the first metric
    df_res = pd.DataFrame.from_dict(results, orient="index").round(4)
    df_res = df_res.sort_values(by=df_res.columns[0], ascending=True)

    return df_res


#
# Other chronos interface due to bug in sktime's Chronos2Forecaster.
#


def evaluate_chronos2_native(y, X_covariates, cv, pipeline, prediction_length, use_covariates=True, ctx_length = 24 * 7 * 4 * 4 * 2,label="Chronos2"):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

    records = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(y)):
        y_train = y.iloc[train_idx]
        y_test  = y.iloc[test_idx]
        X_train = X_covariates.iloc[train_idx]
        X_test  = X_covariates.iloc[test_idx]

        context_df = pd.DataFrame({
            "id":        ["ts0"] * len(y_train),   
            "timestamp": y_train.index,
            "target":    y_train.values,
            **{col: X_train[col].values for col in X_train.columns},
        })

        future_df = pd.DataFrame({
            "id":        ["ts0"] * len(y_test),   
            "timestamp": y_test.index,
            **{col: X_test[col].values for col in X_test.columns},
        })

        # get the last 16 weeks of the training data as context and the next 24 hours as future for prediction
        context_df = context_df.iloc[-ctx_length:]  
        future_df = future_df.iloc[:prediction_length] 

        if use_covariates:  
            pred_df = pipeline.predict_df(
                context_df,
                future_df=future_df,
                prediction_length=prediction_length,
                quantile_levels=[0.1, 0.5, 0.9],
                id_column="id",              
                timestamp_column="timestamp",
                target="target",
            )
        else:
            pred_df = pipeline.predict_df(
                    context_df[["id", "timestamp", "target"]],  # only use the target column for context
                    prediction_length=prediction_length,
                    future_df=None,  # no future covariates
                    quantile_levels=[0.1, 0.5, 0.9],
                    id_column="id",              
                    timestamp_column="timestamp",
                    target="target",
                )



        y_pred = pred_df.set_index("timestamp")["0.5"]  
        y_pred.index = y_test.index

        mse = mean_squared_error(y_test.values, y_pred.values)
        mae = mean_absolute_error(y_test.values, y_pred.values)
        mape = mean_absolute_percentage_error(y_test.values, y_pred.values)


        records.append({
            "fold":                    fold_idx,
            "y_test":                  y_test,
            "y_pred":                  y_pred,
            "test_MeanSquaredError":   mse,
            "test_MeanAbsoluteError":  mae,
            "test_MeanAbsolutePercentageError":              mape,
        })

    df_result = pd.DataFrame(records)
    
    return df_result




#
#
#
# Plots for debugging
#
#
#

def plot_daily_pattern(models_dict, hour_freq=1):
    """
    Average daily profile (averaged across all days) for y_test and all models.
    Shows mean + ±1 standard deviation band.

    Args:
        models_dict (dict): Keys = model names, Values = DataFrames with 'y_pred' and 'y_test'.
        hour_freq (int): Hourly resolution of the x-axis (default=1 → every hour).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    first_df = next(iter(models_dict.values()))

    # ── Ground truth daily profile ────────────────────────────────────────────
    # Stack all days into one array: shape (n_days, horizon)
    gt_days = np.stack([np.array(row) for row in first_df["y_test"].values])
    gt_mean = gt_days.mean(axis=0)
    gt_std  = gt_days.std(axis=0)

    # x-axis: hours of the day (derived from first y_test row)
    sample_index = first_df.iloc[0]["y_test"].index
    try:
        hours = sample_index.hour + sample_index.minute / 60
    except AttributeError:
        hours = np.arange(len(gt_mean))

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(hours, gt_mean, label="Ground Truth", linewidth=2.5, color="black", zorder=5)
    ax.fill_between(hours, gt_mean - gt_std, gt_mean + gt_std,
                    alpha=0.12, color="black")

    # ── Encoding predictions ──────────────────────────────────────────────────
    colors = plt.cm.tab10.colors
    for i, (label, df) in enumerate(models_dict.items()):
        pred_days = np.stack([np.array(row) for row in df["y_pred"].values])
        pred_mean = pred_days.mean(axis=0)
        pred_std  = pred_days.std(axis=0)

        c = colors[i % len(colors)]
        ax.plot(hours, pred_mean, label=label, linewidth=1.8,
                linestyle="--", color=c)
        ax.fill_between(hours, pred_mean - pred_std, pred_mean + pred_std,
                        alpha=0.08, color=c)

    ax.set_title("Average daily profile - Chronos2 encoding comparison",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Time of day (h)")
    ax.set_ylabel("Power")
    ax.set_xticks(np.arange(0, 25, hour_freq))
    ax.set_xticklabels([f"{int(h):02d}:00" for h in np.arange(0, 25, hour_freq)],
                       rotation=45, ha="right")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.show()


def plot_daily_pattern_by_weekday(models_dict):
    """
    Average daily profile separated by weekday (Mon–Sun) - 7 subplots.
    Shows Ground Truth + all encoding variants.

    Args:
        models_dict (dict): Keys = model names, Values = DataFrames with 'y_pred' and 'y_test'.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    weekday_names = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
    first_df = next(iter(models_dict.values()))

    # Weekday assignment from the first index value of each day
    weekdays = np.array([
        first_df.iloc[i]["y_test"].index[0].weekday()
        for i in range(len(first_df))
    ])

    sample_index = first_df.iloc[0]["y_test"].index
    try:
        hours = sample_index.hour + sample_index.minute / 60
    except AttributeError:
        hours = np.arange(len(sample_index))

    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 7, figsize=(22, 4), sharey=True)

    for wd in range(7):
        ax = axes[wd]
        mask = weekdays == wd

        if mask.sum() == 0:
            ax.set_title(weekday_names[wd])
            continue

        # Ground Truth
        gt_days = np.stack([np.array(first_df.iloc[i]["y_test"])
                            for i in np.where(mask)[0]])
        gt_mean = gt_days.mean(axis=0)
        gt_std  = gt_days.std(axis=0)
        ax.plot(hours, gt_mean, color="black", linewidth=2, label="Ground Truth")
        ax.fill_between(hours, gt_mean - gt_std, gt_mean + gt_std,
                        alpha=0.10, color="black")

        # Encodings
        for i, (label, df) in enumerate(models_dict.items()):
            pred_days = np.stack([np.array(df.iloc[j]["y_pred"])
                                  for j in np.where(mask)[0]])
            pred_mean = pred_days.mean(axis=0)
            c = colors[i % len(colors)]
            ax.plot(hours, pred_mean, linestyle="--", linewidth=1.5,
                    color=c, label=label)

        ax.set_title(weekday_names[wd], fontsize=12, fontweight="bold")
        ax.set_xlabel("h")
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Power")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models_dict) + 1,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.18))
    fig.suptitle("Daily profile by weekday - Chronos2 encoding comparison",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_daily_pattern_with_holidays(models_dict, X_full, hour_freq=2):
    """
    Average daily profile with holidays vs. workdays highlighted.

    Shows three layers:
      - Ground Truth (black)
      - Encoding predictions (colored dashed lines)
      - Separate profile for holidays only (colored, semi-transparent)

    Args:
        models_dict (dict): Keys = model names, Values = DataFrames with 'y_pred' and 'y_test'.
        X_full (pd.DataFrame): Must contain a 'holiday' (0/1) column and a DatetimeIndex.
        hour_freq (int): Step size for x-axis ticks in hours.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    first_df = next(iter(models_dict.values()))

    # ── Holiday mask per day ──────────────────────────────────────────────────
    # For each forecast day: is the first timestamp a holiday?
    is_holiday = np.array([
        bool(X_full.loc[first_df.iloc[i]["y_test"].index[0], "holiday"])
        for i in range(len(first_df))
    ])
    is_workday = ~is_holiday

    # ── x-axis from the first y_test index ────────────────────────────────────
    sample_index = first_df.iloc[0]["y_test"].index
    try:
        hours = sample_index.hour + sample_index.minute / 60
    except AttributeError:
        hours = np.arange(len(sample_index))

    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    titles    = ["Workdays / Normal days", "Holidays / Holidays"]
    masks     = [is_workday, is_holiday]

    for ax, title, mask in zip(axes, titles, masks):
        n = mask.sum()
        ax.set_title(f"{title}  (n={n} days)", fontsize=13, fontweight="bold")

        if n == 0:
            ax.text(0.5, 0.5, "No days", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_xlabel("Time of day (h)")
            continue

        # Ground Truth
        gt_days  = np.stack([np.array(first_df.iloc[i]["y_test"])
                             for i in np.where(mask)[0]])
        gt_mean  = gt_days.mean(axis=0)
        gt_std   = gt_days.std(axis=0)
        ax.plot(hours, gt_mean, color="black", linewidth=2.5,
                label="Ground Truth", zorder=5)
        ax.fill_between(hours, gt_mean - gt_std, gt_mean + gt_std,
                        alpha=0.10, color="black")

        # Encoding predictions
        for i, (label, df) in enumerate(models_dict.items()):
            pred_days = np.stack([np.array(df.iloc[j]["y_pred"])
                                  for j in np.where(mask)[0]])
            pred_mean = pred_days.mean(axis=0)
            pred_std  = pred_days.std(axis=0)
            c = colors[i % len(colors)]
            ax.plot(hours, pred_mean, linestyle="--", linewidth=1.8,
                    color=c, label=label)
            ax.fill_between(hours, pred_mean - pred_std, pred_mean + pred_std,
                            alpha=0.07, color=c)

        # Holiday panel: plot individual days as thin lines in the background
        if mask is is_holiday:
            for j in np.where(mask)[0]:
                gt_single = np.array(first_df.iloc[j]["y_test"])
                ax.plot(hours, gt_single, color="gray", linewidth=0.6,
                        alpha=0.35, zorder=1)

        ax.set_xlabel("Time of day (h)")
        ax.set_xticks(np.arange(0, 25, hour_freq))
        ax.set_xticklabels(
            [f"{int(h):02d}:00" for h in np.arange(0, 25, hour_freq)],
            rotation=45, ha="right"
        )
        ax.grid(True, alpha=0.30)

    axes[0].set_ylabel("Power")

    # ── Shared legend ────────────────────────────────────────────────────────
    handles, labels = axes[0].get_legend_handles_labels()
    # Extra legend entry for individual holidays
    gray_patch = mpatches.Patch(color="gray", alpha=0.5,
                                label="Individual holidays (GT)")
    fig.legend(
        handles + [gray_patch],
        labels  + ["Individual holidays (GT)"],
        loc="lower center", ncol=len(models_dict) + 2,
        fontsize=10, framealpha=0.95,
        bbox_to_anchor=(0.5, -0.22),
    )
    fig.suptitle(
        "Daily profile: Workdays vs. Holidays - Chronos2 encoding comparison",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.show()

def plot_full_series_with_holidays(models_dict, X_full):
    """
    Full time series with colored holiday bands in the background.

    Args:
        models_dict (dict): Keys = model names, Values = DataFrames with
                            'y_pred' and 'y_test'.
        X_full (pd.DataFrame): Must contain a 'holiday' (0/1) column and a DatetimeIndex.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import pandas as pd
    import numpy as np

    first_df = next(iter(models_dict.values()))

    # ── Full GT time series ───────────────────────────────────────────────────
    y_true_series = pd.concat(first_df["y_test"].values)

    # ── Determine holiday intervals ───────────────────────────────────────────
    # Consecutive holiday blocks as (start, end) tuples
    holiday_mask = X_full.loc[y_true_series.index, "holiday"].astype(bool)
    holiday_blocks = []
    in_block = False
    for ts, is_hol in holiday_mask.items():
        if is_hol and not in_block:
            block_start = ts
            in_block = True
        elif not is_hol and in_block:
            holiday_blocks.append((block_start, ts))
            in_block = False
    if in_block:  # last block
        holiday_blocks.append((block_start, holiday_mask.index[-1]))

    # ── Plot ─────────────────────────────────────────────────────────────────
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(16, 5))

    # Holiday bands first (background)
    for start, end in holiday_blocks:
        ax.axvspan(start, end, color="#FFD700", alpha=0.25, zorder=0)

    # Ground Truth
    ax.plot(y_true_series.index, y_true_series.values,
            label="Ground Truth", color="black", linewidth=1.8, zorder=5)

    # Encoding predictions
    for i, (label, df) in enumerate(models_dict.items()):
        y_pred_series = pd.concat(df["y_pred"].values)
        ax.plot(y_pred_series.index, y_pred_series.values,
                label=label, linestyle="--", linewidth=1.3,
                color=colors[i % len(colors)], zorder=4)

    # Holiday legend manually
    holiday_patch = mpatches.Patch(
        color="#FFD700", alpha=0.5, label="Holiday"
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles + [holiday_patch],
        labels  + ["Holiday"],
        loc="upper left", fontsize=10, framealpha=0.95,
        ncol=min(len(models_dict) + 2, 4),
    )

    ax.set_title(
        "Full time series with holiday markers - Chronos2 encodings",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Power")
    ax.grid(True, alpha=0.30)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


#
#
#

# ══════════════════════════════════════════════════════════════════════════════
#  Feature Encoding Visualizations
#  Functions: plot_hour_encodings_over_time
#             plot_holiday_encodings_over_time
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── Shared helpers ────────────────────────────────────────────────────────────

def _encoding_style(ax, label, color, ylabel=None, rotate_x=0):
    """Shared panel style for encoding plots."""
    surf = "#FFFFFF"
    ax.set_facecolor(surf)
    ax.set_ylabel(ylabel or "", fontsize=8, color="#797876", labelpad=6)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#d4d1ca")
    ax.grid(True, axis="y", alpha=0.5)
    ax.tick_params(axis="x", which="both", length=0, labelrotation=rotate_x)
    if rotate_x:
        plt.setp(ax.get_xticklabels(), ha="right")  # Rechtsbündig bei Schrägstellung
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
    _encoding_style(ax, "Target  (holiday vs. dow)", gold, ylabel="target value", xlabel="Date", rotate_x=45)
    

    return fig



class EncodingFeatureBuilder:
    """Builds all feature encoding variants for a given target series."""

    def __init__(
        self,
        y: pd.Series,
        cv,
        prediction_length: int,
        cyclic_encode_fn,
        scope: str = "target",
    ):
        self.y                  = y
        self.cv                 = cv
        self.prediction_length  = prediction_length
        self.cyclic_encode_fn   = cyclic_encode_fn
        self.scope              = scope

    def build(self) -> dict[str, pd.DataFrame]:
        X_full  = self._build_raw()
        y_train, X_train = self._first_fold(X_full)
        return {
            "X_full":   X_full,
            "X_onehot": self._onehot(X_full),
            "X_target": self._target(X_full, y_train, X_train),
            "X_cyclic": self._cyclic(X_full),
            "X_scaled": self._scaled(X_full, y_train),
        }

    # ── Raw features ──────────────────────────────────────────────────────────
    def _build_raw(self) -> pd.DataFrame:
        full_index = pd.date_range(
            start=self.y.index.min(),
            periods=len(self.y) + self.prediction_length,
            freq="h",
        )
        y_full = pd.DataFrame(index=full_index, data={self.y.name or self.scope: 0.0})

        X_holiday = HolidayFeatures(
            calendar=country_holidays(country="DE", subdiv="BW"),
            include_bridge_days=True,
        ).fit_transform(y_full)
        X_holiday["holiday"] = X_holiday.sum(axis=1)

        X_calendar = DateTimeFeatures(
            ts_freq="H",
            manual_selection=["day_of_week", "hour_of_day"],
        ).fit_transform(y_full)

        return pd.concat([X_holiday["holiday"], X_calendar], axis=1)

    # ── Leak-free first fold ──────────────────────────────────────────────────
    def _first_fold(self, X_full):
        first_train_idx, _ = next(iter(self.cv.split(self.y)))
        y_train = self.y.iloc[first_train_idx]
        return y_train, X_full.loc[y_train.index]

    # ── 1) One-Hot ────────────────────────────────────────────────────────────
    def _onehot(self, X_full) -> pd.DataFrame:
        return pd.concat([
            X_full["holiday"].astype(float),
            pd.get_dummies(X_full["day_of_week"],  prefix="dow",  drop_first=True).astype(float),
            pd.get_dummies(X_full["hour_of_day"], prefix="hour", drop_first=True).astype(float),
        ], axis=1)

    # ── 2) Target Encoding ────────────────────────────────────────────────────
    def _target(self, X_full, y_train, X_train) -> pd.DataFrame:
        dow_enc     = y_train.groupby(X_train["day_of_week"]).mean()
        hour_enc    = y_train.groupby(X_train["hour_of_day"]).mean()
        holiday_enc = y_train.groupby(X_train["holiday"]).mean()

        X_target = pd.DataFrame({
            "holiday_target": X_full["holiday"].map(holiday_enc).fillna(holiday_enc.mean()),
            "dow_target":     X_full["day_of_week"].map(dow_enc).fillna(dow_enc.mean()),
            "hour_target":    X_full["hour_of_day"].map(hour_enc).fillna(hour_enc.mean()),
        }, index=X_full.index)

        for combi_cols, target_col in [
            (["hour_of_day", "day_of_week"], "hour_dow_target"),
            (["hour_of_day", "holiday"],     "hour_holiday_target"),
        ]:
            combi_enc = (
                y_train
                .groupby([X_train[c] for c in combi_cols])
                .mean().rename(target_col).reset_index()
            )
            combi_enc.columns = combi_cols + [target_col]
            X_target[target_col] = (
                X_full[combi_cols]
                .merge(combi_enc, on=combi_cols, how="left")[target_col]
                .values
            )
            X_target[target_col] = X_target[target_col].fillna(X_target[target_col].mean())

        return X_target

    # ── 3) Cyclic Encoding ────────────────────────────────────────────────────
    def _cyclic(self, X_full) -> pd.DataFrame:
        dow_sin,   dow_cos   = self.cyclic_encode_fn(X_full["day_of_week"], max_val=7)
        hour_sin,  hour_cos  = self.cyclic_encode_fn(X_full["hour_of_day"], max_val=24)
        month_sin, month_cos = self.cyclic_encode_fn(
            pd.Series(X_full.index.month, index=X_full.index), max_val=12
        )
        return pd.DataFrame({
            "holiday":   X_full["holiday"].astype(float),
            "dow_sin":   dow_sin.values,   "dow_cos":   dow_cos.values,
            "hour_sin":  hour_sin.values,  "hour_cos":  hour_cos.values,
            "month_sin": month_sin.values, "month_cos": month_cos.values,
        }, index=X_full.index)

    # ── 4) Scaled Encoding ────────────────────────────────────────────────────
    def _scaled(self, X_full, y_train) -> pd.DataFrame:
        scaler = StandardScaler()
        scaler.fit(X_full[["day_of_week", "hour_of_day"]].loc[y_train.index])
        X_scaled = pd.DataFrame(
            scaler.transform(X_full[["day_of_week", "hour_of_day"]]),
            index=X_full.index,
            columns=["day_of_week", "hour_of_day"],
        )
        X_scaled["holiday"] = X_full["holiday"].astype(float).values
        return X_scaled