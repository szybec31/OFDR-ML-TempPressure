"""
Tabele i wykresy generowane z wyników JEDNEGO uruchomienia (folds + predictions).

Konwencje:
- "common folds" = foldy, w których WSZYSTKIE modele mają status ok; na nich liczone jest
  główne porównanie (summary_common), testy statystyczne i współczynnik G,
- średnia i odchylenie standardowe po foldach (ddof=1),
- R^2 per fold jest raportowane tylko tam, gdzie cel zmienia się w foldzie; R^2 "pooled"
  (po złączeniu predykcji ze wszystkich foldów) jest osobną, podpisaną tabelą.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from .metrics import METRIC_NAMES, TARGETS, evaluate

ALPHA = 0.05


# ---------------------------------------------------------------- zapis tabel

def _fmt(v, digits=3):
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.{digits}f}"
    return str(v)


def write_table(df, out_dir, name, title=None, digits=3):
    """Zapisuje tabelę jako CSV (pełna precyzja), Markdown i LaTeX (zaokrąglone)."""
    df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)

    shown = df.copy()
    for c in shown.columns:
        shown[c] = shown[c].map(lambda v: _fmt(v, digits))

    lines = []
    if title:
        lines += [f"### {title}", ""]
    lines.append("| " + " | ".join(shown.columns) + " |")
    lines.append("|" + "|".join(["---"] * len(shown.columns)) + "|")
    for _, r in shown.iterrows():
        lines.append("| " + " | ".join(r.values) + " |")
    with open(os.path.join(out_dir, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    latex = shown.to_latex(index=False, escape=True, caption=title, label=f"tab:{name}")
    with open(os.path.join(out_dir, f"{name}.tex"), "w", encoding="utf-8") as f:
        f.write(latex)


# ---------------------------------------------------------------- podsumowania

def common_folds(folds):
    ok = folds.pivot_table(index="fold", columns="model", values="status",
                           aggfunc=lambda s: (s == "ok").all())
    ok = ok.fillna(False).astype(bool)
    return sorted(ok.index[ok.all(axis=1)].tolist())


def summarize(folds, models, fold_ids=None):
    rows = []
    for m in models:
        f = folds[(folds["model"] == m) & (folds["status"] == "ok")]
        if fold_ids is not None:
            f = f[f["fold"].isin(fold_ids)]
        row = {"model": m, "n_folds": len(f)}
        for k in METRIC_NAMES:
            if k not in f:
                continue
            vals = f[k].dropna()
            row[f"{k}_mean"] = vals.mean() if len(vals) else np.nan
            row[f"{k}_std"] = vals.std(ddof=1) if len(vals) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_metrics(predictions, models, fold_ids):
    rows = []
    test = predictions[(predictions["subset"] == "test") & predictions["fold"].isin(fold_ids)]
    for m in models:
        p = test[test["model"] == m]
        if p.empty:
            continue
        res = evaluate(p[TARGETS], p[["pressure_pred", "dT_pred"]].to_numpy())
        rows.append({"model": m, "n_samples": len(p), **res})
    return pd.DataFrame(rows)


def compact_summary(summary):
    """Tabela do raportu: 'średnia (std)' dla najważniejszych metryk."""
    cols = ["pressure_mae", "pressure_rmse", "pressure_maxae", "pressure_bias",
            "dT_mae", "dT_rmse", "dT_maxae", "dT_bias", "pressure_r2"]
    out = summary[["model", "n_folds"]].copy()
    for c in cols:
        if f"{c}_mean" in summary:
            out[c] = [
                f"{_fmt(m)} ({_fmt(s)})" if not np.isnan(m) else "N/A"
                for m, s in zip(summary[f"{c}_mean"], summary[f"{c}_std"])
            ]
    return out


# ---------------------------------------------------------------- testy i zysk G

def paired_tests(folds, baseline, models, fold_ids, metrics=("pressure_rmse", "dT_rmse")):
    rows = []
    base = folds[(folds["model"] == baseline) & folds["fold"].isin(fold_ids)].set_index("fold")
    n = len(fold_ids)
    min_p = 2.0 / 2 ** n if n > 0 else np.nan
    for m in models:
        if m == baseline:
            continue
        other = folds[(folds["model"] == m) & folds["fold"].isin(fold_ids)].set_index("fold")
        for metric in metrics:
            a = base.loc[fold_ids, metric].to_numpy(dtype=float)
            b = other.loc[fold_ids, metric].to_numpy(dtype=float)
            d = a - b
            if n < 2 or np.allclose(d, 0):
                w_stat, w_p, t_p, cohen = np.nan, 1.0, 1.0, 0.0
            else:
                w_stat, w_p = wilcoxon(a, b)
                t_p = ttest_rel(a, b).pvalue
                cohen = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
            rows.append({
                "baseline": baseline, "model": m, "metric": metric, "n_folds": n,
                "baseline_mean": a.mean(), "model_mean": b.mean(),
                "mean_diff_baseline_minus_model": d.mean(),
                "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
                "min_attainable_p": min_p,
                "ttest_p": t_p, "cohens_d": cohen,
                "significant": bool(w_p < ALPHA),
                "better_by_mean": m if d.mean() > 0 else (baseline if d.mean() < 0 else "tie"),
            })
    return pd.DataFrame(rows)


def gain_table(summary, baseline):
    """G = (RMSE_base - RMSE_model) / RMSE_base * 100% (Załącznik, krok 4), na tych samych foldach."""
    s = summary.set_index("model")
    if baseline not in s.index:
        return pd.DataFrame()
    rows = []
    for m in s.index:
        if m == baseline:
            continue
        row = {"model": m}
        for t in TARGETS:
            b, v = s.loc[baseline, f"{t}_rmse_mean"], s.loc[m, f"{t}_rmse_mean"]
            row[f"{t}_rmse_baseline"] = b
            row[f"{t}_rmse_model"] = v
            row[f"G_{t}_percent"] = (b - v) / b * 100 if b else np.nan
            row[f"G_{t}_ge_15"] = bool(row[f"G_{t}_percent"] >= 15)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- powtarzalność

def signal_drift(df):
    """Dryft sygnału zero_end względem zero_start z tej samej serii i tego samego powtórzenia."""
    start = df[df["point_type"] == "zero_start"].set_index("sample_id")
    end = df[df["point_type"] == "zero_end"]
    rows = []
    for _, e in end.iterrows():
        if e["ref"] not in start.index:
            continue
        s = start.loc[e["ref"]]
        dx, dy = e["mu_X"] - s["mu_X"], e["mu_Y"] - s["mu_Y"]
        rows.append({
            "sample_id": e["sample_id"], "series_id": e["series_id"], "dT": e["dT"],
            "rep_idx": e["rep_idx"],
            "delta_mu_X": dx, "delta_mu_Y": dy,
            "delta_m": (dx + dy) / 2,       # składowa wspólna (mean_XY)
            "delta_d": dx - dy,             # składowa różnicowa (diff_XY)
            "delta_mu_norm": np.hypot(dx, dy),
        })
    return pd.DataFrame(rows)


def residual_drift(predictions, drift):
    """
    RD = sqrt((T_hat_end - T_start)^2 + (p_hat_end - 0)^2) z predykcji OUT-OF-FOLD
    (model nie widział temperatury, z której pochodzi zero_end). Składowe raportowane osobno,
    bo RD miesza °C i MPa.
    """
    z = predictions[predictions["subset"] == "zero_end"].copy()
    if z.empty:
        return pd.DataFrame()
    z["err_dT"] = z["dT_pred"] - z["dT"]
    z["err_pressure"] = z["pressure_pred"] - 0.0
    z["RD"] = np.hypot(z["err_dT"], z["err_pressure"])
    if not drift.empty:
        z = z.merge(drift[["sample_id", "delta_m", "delta_d"]], on="sample_id", how="left")
    return z


def residual_drift_summary(rd):
    if rd.empty:
        return pd.DataFrame()
    agg = rd.groupby("model").agg(
        n=("RD", "size"),
        RD_mean=("RD", "mean"), RD_median=("RD", "median"), RD_max=("RD", "max"),
        err_dT_mean=("err_dT", "mean"), err_pressure_mean=("err_pressure", "mean"),
        abs_err_dT_mean=("err_dT", lambda s: s.abs().mean()),
        abs_err_pressure_mean=("err_pressure", lambda s: s.abs().mean()),
    )
    return agg.reset_index()


# ---------------------------------------------------------------- wykresy

def plot_rmse_bars(summary, out_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, t, unit in zip(axes, TARGETS, ["MPa", "°C"]):
        s = summary.dropna(subset=[f"{t}_rmse_mean"])
        ax.bar(s["model"], s[f"{t}_rmse_mean"], yerr=s[f"{t}_rmse_std"], capsize=3, color="#4C72B0")
        ax.set_yscale("log")
        ax.set_title(f"RMSE {t} [{unit}]")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pred_vs_true(predictions, models, fold_ids, out_path):
    test = predictions[(predictions["subset"] == "test") & predictions["fold"].isin(fold_ids)]
    models = [m for m in models if m in set(test["model"])]
    if not models:
        return
    fig, axes = plt.subplots(len(models), 2, figsize=(9, 3.2 * len(models)), squeeze=False)
    for i, m in enumerate(models):
        p = test[test["model"] == m]
        for j, t in enumerate(TARGETS):
            ax = axes[i, j]
            ax.scatter(p[t], p[f"{t}_pred"], s=6, alpha=0.5)
            lo, hi = p[t].min(), p[t].max()
            ax.plot([lo, hi], [lo, hi], color="k", lw=0.8)
            ax.set_title(f"{m}: {t}")
            ax.set_xlabel("prawdziwe")
            ax.set_ylabel("przewidziane")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_residual_hist(predictions, models, fold_ids, out_path):
    test = predictions[(predictions["subset"] == "test") & predictions["fold"].isin(fold_ids)]
    models = [m for m in models if m in set(test["model"])]
    if not models:
        return
    fig, axes = plt.subplots(len(models), 2, figsize=(9, 2.8 * len(models)), squeeze=False)
    for i, m in enumerate(models):
        p = test[test["model"] == m]
        for j, t in enumerate(TARGETS):
            axes[i, j].hist(p[f"{t}_pred"] - p[t], bins=40)
            axes[i, j].set_title(f"{m}: reszty {t} (pred - true)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- całość

def build_report(result, df, cfg, out_dir):
    folds, preds = result["folds"], result["predictions"]
    models, baseline = cfg["models"], cfg.get("baseline")
    folds.to_csv(os.path.join(out_dir, "folds.csv"), index=False)
    preds.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    if not result["coefficients"].empty:
        write_table(result["coefficients"], out_dir, "physical_coefficients", "Współczynniki modeli analitycznych per fold", 6)

    all_folds = sorted(folds["fold"].unique().tolist())
    common = common_folds(folds)
    excluded = sorted(set(all_folds) - set(common))
    status = folds.pivot_table(index=["fold", "test_group"], columns="model", values="status", aggfunc="first").reset_index()
    write_table(status, out_dir, "fold_status", "Status modeli w foldach (ok / not_identifiable)")

    summary_common = summarize(folds, models, common)
    summary_all = summarize(folds, models, None)
    write_table(summary_common, out_dir, "summary_common_folds")
    write_table(summary_all, out_dir, "summary_all_valid_folds")
    write_table(compact_summary(summary_common), out_dir, "results_table",
                f"Wyniki na wspólnych foldach ({len(common)}/{len(all_folds)}), średnia (std) po foldach")
    write_table(pooled_metrics(preds, models, common), out_dir, "pooled_metrics",
                "Metryki pooled (predykcje ze wspólnych foldów złączone), w tym R^2 globalne")

    if baseline and baseline in models and len(common) >= 2:
        write_table(paired_tests(folds, baseline, models, common), out_dir, "paired_tests",
                    f"Testy parowane vs {baseline} (wspólne foldy, n={len(common)})", 4)
        write_table(gain_table(summary_common, baseline), out_dir, "gain_vs_baseline",
                    f"Współczynnik zysku G względem {baseline} (RMSE, wspólne foldy)", 2)

    drift = signal_drift(df)
    if not drift.empty:
        write_table(drift, out_dir, "repeatability_signal_drift", digits=4)
        by_dt = drift.groupby("dT").agg(
            n=("delta_mu_X", "size"),
            delta_mu_X=("delta_mu_X", "mean"), delta_mu_Y=("delta_mu_Y", "mean"),
            delta_m=("delta_m", "mean"), delta_d=("delta_d", "mean"),
            delta_mu_norm=("delta_mu_norm", "mean"),
        ).reset_index()
        write_table(by_dt, out_dir, "repeatability_signal_drift_by_dT", "Dryft sygnału zero_end - zero_start wg dT", 4)
    rd = residual_drift(preds, drift)
    if not rd.empty:
        rd.to_csv(os.path.join(out_dir, "repeatability_residual_drift.csv"), index=False)
        write_table(residual_drift_summary(rd), out_dir, "repeatability_residual_drift_summary",
                    "Residual Drift na zero_end (predykcje out-of-fold)", 4)

    write_table(result["split_diagnostics"], out_dir, "split_diagnostics", "Diagnostyka podziałów (wspólne serie train/test)")

    plot_rmse_bars(summary_common, os.path.join(out_dir, "rmse_common_folds.png"),
                   f"{cfg['name']}: RMSE na wspólnych foldach")
    plot_pred_vs_true(preds, models, common, os.path.join(out_dir, "pred_vs_true.png"))
    plot_residual_hist(preds, models, common, os.path.join(out_dir, "residuals_hist.png"))

    return {"common_folds": common, "excluded_folds": excluded, "summary_common": summary_common}
