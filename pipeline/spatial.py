"""
Punkt 7: wykorzystanie przestrzennego charakteru pomiarów OFDR.

1. Cache: każdy plik (X i Y) jest czyszczony tak samo jak w build_folder_summary (clean_roi: ROI,
   obcięcie brzegów, odrzucenie outlierów względem mediany CAŁEGO ROI), a następnie punkty są
   sumowane w binach o długości base_bin_cm wzdłuż włókna (po położeniu, nie po indeksie - pliki
   mają różne siatki). Okno o długości L = k * base_bin_cm to suma k sąsiednich binów, więc średnia
   w oknie jest liczona dokładnie. Okno "full" = całe ROI (= mu_X, mu_Y z paired_features).

2. Dla każdej długości L i każdego okna: cechy X, Y, m, d; ocena w zewnętrznej walidacji
   (domyślnie leave-one-temperature-out) dwóch modeli:
       AN-JOINT - model fizyczny  mu_k = A_k dT^2 + B_k dT + C_k p  (kalibracja łączna MNK),
       CD-LIN   - regresja liniowa (p, dT) na [1, m, d],
   w dwóch trybach kalibracji:
       local  - osobne współczynniki dla każdego okna,
       global - jeden zestaw współczynników dla wszystkich okien (MNK na złączonych oknach).
   RMSE(L) = RMSE po wszystkich (oknach x rekordach testowych) folda, uśrednione po foldach.

   Okna, w których po czyszczeniu zostało < min_coverage typowej liczby punktów, są pomijane (NaN).
   Uwaga: na brzegach ROI (z ~ 8.5 m i ~ 12 m) włókno wchodzi / wychodzi z komory - sygnał przechodzi
   do zera, a clean_roi wycina tam większość punktów jako outliery.

3. Lokalne czułości: AN-JOINT dopasowany na wszystkich rekordach dla okien o wybranych L ->
   profile A, B, C wzdłuż włókna, ich jednorodność oraz okna z systematycznie większym błędem.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from baselines.models.physical_model import invert_quadratic
from utils import ROI_MAX, ROI_MIN, clean_roi, read_measurement_file
from .report import write_table
from .metrics import TARGET_UNITS, get_targets
from .splits import select_folds


# ---------------------------------------------------------------- cache binów

def _bin_file(path, edges):
    roi, _, _ = clean_roi(read_measurement_file(path))
    nb = len(edges) - 1
    if roi is None:
        return np.zeros(nb), np.zeros(nb)
    z, v = roi["length"].to_numpy(), roi["value"].to_numpy()
    idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, nb - 1)
    return np.bincount(idx, weights=v, minlength=nb), np.bincount(idx, minlength=nb).astype(float)


def build_bin_cache(cfg, df):
    """Sumy i liczności punktów w binach dla kanałów X i Y; wiersze w kolejności df['sample_id']."""
    base_m = cfg["base_bin_cm"] / 100
    nb = int(round((ROI_MAX - ROI_MIN) / base_m))
    edges = ROI_MIN + base_m * np.arange(nb + 1)
    path = cfg["bin_cache"].format(base=cfg["base_bin_cm"])

    if os.path.exists(path):
        c = np.load(path, allow_pickle=True)
        if list(c["sample_id"]) == list(df["sample_id"]) and np.allclose(c["edges"], edges):
            return {k: c[k] for k in c.files}
        print("Cache binów nieaktualny - przeliczam.")

    paths = pd.read_csv(cfg["training_dataset"]).set_index("sample_id")["path_x"]
    path_x = df["sample_id"].map(paths)
    if path_x.isna().any():
        raise RuntimeError("Brak path_x dla części próbek - uruchom ponownie `python main.py setup`.")

    files = list(df["sample_id"]) + list(path_x)
    out = Parallel(n_jobs=cfg.get("n_jobs", -1), return_as="generator")(
        delayed(_bin_file)(f, edges) for f in files)
    res = list(tqdm(out, total=len(files), desc="Biny przestrzenne"))
    n = len(df)
    sums = np.array([r[0] for r in res])
    cnts = np.array([r[1] for r in res])
    cache = {
        "sample_id": np.array(df["sample_id"]), "edges": edges,
        "sum_Y": sums[:n], "cnt_Y": cnts[:n], "sum_X": sums[n:], "cnt_X": cnts[n:],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **cache)
    return cache


def window_signals(cache, L_cm, base_cm, min_coverage):
    """Średnie mu_X, mu_Y w niepokrywających się oknach długości L (lub 'full'); okna od ROI_MIN."""
    nb = cache["sum_X"].shape[1]
    k = nb if L_cm == "full" else int(round(L_cm / base_cm))
    if k < 1 or k > nb:
        raise ValueError(f"Długość okna {L_cm} cm poza zakresem ROI")
    nw = nb // k
    edges = cache["edges"]
    centers = np.array([(edges[i * k] + edges[(i + 1) * k]) / 2 for i in range(nw)])

    out = {"z": centers, "n_windows": nw, "bins_used": nw * k}
    for ch in ["X", "Y"]:
        # pokrycie liczone względem typowej liczby punktów w binie danego kanału
        expected = np.median(cache[f"cnt_{ch}"][cache[f"cnt_{ch}"] > 0]) * k
        s = cache[f"sum_{ch}"][:, : nw * k].reshape(len(cache["sum_X"]), nw, k).sum(axis=2)
        c = cache[f"cnt_{ch}"][:, : nw * k].reshape(len(cache["sum_X"]), nw, k).sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = s / c
        mu[c < min_coverage * expected] = np.nan
        out[ch] = mu
    return out


# ---------------------------------------------------------------- modele (numpy, per okno)

def _lstsq(F, S):
    """MNK z pominięciem wierszy z NaN; F (n, k), S (n,) lub (n, q). Zwraca None, gdy osobliwe."""
    ok = np.all(np.isfinite(np.column_stack([F, S])), axis=1)
    if ok.sum() <= F.shape[1] or np.linalg.matrix_rank(F[ok]) < F.shape[1]:
        return None
    return np.linalg.lstsq(F[ok], S[ok], rcond=None)[0]


def _phys_design(y):
    p, T = y[:, 0], y[:, 1]
    return np.column_stack([T ** 2, T, p])


def fit_an_joint(sx, sy, y):
    F = _phys_design(y)
    cx, cy = _lstsq(F, sx), _lstsq(F, sy)
    if cx is None or cy is None:
        return None
    return {"Ax": cx[0], "Bx": cx[1], "Cx": cx[2], "Ay": cy[0], "By": cy[1], "Cy": cy[2],
            "t_mid": 0.5 * (y[:, 1].min() + y[:, 1].max())}


def predict_an_joint(c, sx, sy):
    p, T, _ = invert_quadratic(sx, sy, c["Ax"], c["Bx"], c["Cx"], c["Ay"], c["By"], c["Cy"], c["t_mid"])
    return p, T


def fit_cd_lin(sx, sy, y):
    m, d = (sx + sy) / 2, sx - sy
    return _lstsq(np.column_stack([np.ones_like(m), m, d]), y)


def predict_cd_lin(c, sx, sy):
    m, d = (sx + sy) / 2, sx - sy
    pred = c[0] + np.multiply.outer(m, c[1]) + np.multiply.outer(d, c[2])
    return pred[..., 0], pred[..., 1]


MODELS = {"AN-JOINT": (fit_an_joint, predict_an_joint), "CD-LIN": (fit_cd_lin, predict_cd_lin)}


def _fold_errors(model, mode, sx, sy, y_tr, tr, te, y_te):
    """Błędy (n_test, W) dla p i T; NaN tam, gdzie okno nie dało się skalibrować / brak sygnału."""
    fit, predict = MODELS[model]
    W = sx.shape[1]
    ep = np.full((te.sum(), W), np.nan)
    et = np.full((te.sum(), W), np.nan)
    if mode == "global":
        c = fit(sx[tr].ravel(), sy[tr].ravel(), np.repeat(y_tr, W, axis=0))
        if c is None:
            return ep, et
        p, T = predict(c, sx[te], sy[te])
        return p - y_te[:, :1], T - y_te[:, 1:]
    for w in range(W):
        c = fit(sx[tr, w], sy[tr, w], y_tr)
        if c is None:
            continue
        p, T = predict(c, sx[te, w], sy[te, w])
        ep[:, w], et[:, w] = p - y_te[:, 0], T - y_te[:, 1]
    return ep, et


# ---------------------------------------------------------------- analiza

def _rmse(e):
    return float(np.sqrt(np.nanmean(e ** 2))) if np.isfinite(e).any() else np.nan


def _min_length(table, limits, thresholds, targets):
    """Najmniejsze L, od którego (wraz z dłuższymi oknami) RMSE spełnia kryterium."""
    rows = []
    for (model, mode), g in table.groupby(["model", "mode"], sort=False):
        g = g.sort_values("L_m", ascending=False)
        full = g[g["L_label"] == "full"].iloc[0]
        p_rmse, t_rmse = (f"{t}_rmse" for t in targets)
        lim_p, lim_t = limits["pressure"], limits["T"]
        crits = {f"abs_p<={lim_p}": g[p_rmse] <= lim_p, f"abs_T<={lim_t}": g[t_rmse] <= lim_t}
        crits["abs_both"] = crits[f"abs_p<={lim_p}"] & crits[f"abs_T<={lim_t}"]
        for x in thresholds:
            crits[f"rel_both<=+{int(round(x * 100))}%"] = (
                (g[p_rmse] <= (1 + x) * full[p_rmse]) & (g[t_rmse] <= (1 + x) * full[t_rmse]))
        row = {"model": model, "mode": mode}
        for name, ok in crits.items():
            ok = ok.to_numpy()
            k = len(ok) if ok.all() else int(np.argmin(ok))
            row[f"L_min_cm[{name}]"] = g["L_cm"].iloc[k - 1] if k > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _sensitivities(sig, y):
    """AN-JOINT na wszystkich rekordach, osobno dla każdego okna: współczynniki i ich błędy std."""
    F = _phys_design(y)
    rows = []
    t_mid = 0.5 * (y[:, 1].min() + y[:, 1].max())
    for w in range(sig["X"].shape[1]):
        row = {"window": w, "z_m": sig["z"][w]}
        for ch in ["X", "Y"]:
            s = sig[ch][:, w]
            ok = np.isfinite(s)
            Fo, so = F[ok], s[ok]
            if ok.sum() < 10:
                for name in "ABC":
                    row[f"{name}_{ch}"] = row[f"se_{name}_{ch}"] = np.nan
                row[f"ST_{ch}"] = row[f"rmse_fit_{ch}"] = np.nan
                continue
            coef = np.linalg.lstsq(Fo, so, rcond=None)[0]
            resid = so - Fo @ coef
            cov = np.linalg.inv(Fo.T @ Fo) * resid.var(ddof=3)
            se = np.sqrt(np.diag(cov))
            for i, name in enumerate("ABC"):
                row[f"{name}_{ch}"], row[f"se_{name}_{ch}"] = coef[i], se[i]
            # czułość temperaturowa w środku zakresu: d(mu)/dT = B + 2 A T_mid
            row[f"ST_{ch}"] = coef[1] + 2 * coef[0] * t_mid
            row[f"rmse_fit_{ch}"] = float(np.sqrt(np.mean(resid ** 2)))
        row["C_d"] = row["C_X"] - row["C_Y"]
        row["ST_m"] = 0.5 * (row["ST_X"] + row["ST_Y"])
        rows.append(row)
    return pd.DataFrame(rows)


def _homogeneity(sens):
    rows = []
    for col in ["A_X", "B_X", "C_X", "A_Y", "B_Y", "C_Y", "ST_X", "ST_Y", "C_d", "ST_m"]:
        v = sens[col]
        se = sens.get(f"se_{col}")
        rows.append({
            "coef": col, "mean": v.mean(), "std": v.std(ddof=1),
            "cv_pct": 100 * v.std(ddof=1) / abs(v.mean()) if v.mean() else np.nan,
            "min": v.min(), "max": v.max(),
            "median_se": se.median() if se is not None else np.nan,
            # >> 1: rozrzut między oknami znacznie większy niż niepewność dopasowania w oknie
            "std_over_se": v.std(ddof=1) / se.median() if se is not None else np.nan,
        })
    return pd.DataFrame(rows)


def _outlier_windows(prof):
    rows = []
    for (L, model, mode, t), g in prof.groupby(["L_cm", "model", "mode", "target"]):
        g = g.dropna(subset=["rmse"])
        med = g["rmse"].median()
        mad = 1.4826 * (g["rmse"] - med).abs().median()
        thr = med + 3 * mad
        bad = g[g["rmse"] > thr]
        for _, r in bad.iterrows():
            rows.append({"L_cm": L, "model": model, "mode": mode, "target": t, "window": r["window"],
                         "z_m": r["z_m"], "rmse": r["rmse"], "median_rmse": med, "threshold": thr})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- wykresy

def _plot_rmse_vs_L(table, out_path, targets, stat="mean"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, t in zip(axes, targets):
        unit = TARGET_UNITS[t]
        for (model, mode), g in table.groupby(["model", "mode"], sort=False):
            g = g.sort_values("L_m")
            ls = "-" if mode == "local" else "--"
            if stat == "mean":
                ax.errorbar(g["L_m"] * 100, g[f"{t}_rmse"], yerr=g[f"{t}_rmse_std"], marker="o", ms=4,
                            capsize=3, ls=ls, label=f"{model} ({mode})")
            else:
                ax.plot(g["L_m"] * 100, g[f"{t}_rmse_wmedian"], marker="o", ms=4, ls=ls, label=f"{model} ({mode})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("długość okna L [cm]")
        ax.set_ylabel(f"RMSE {'p' if t == 'pressure' else t} [{unit}]")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("RMSE w funkcji długości okna uśredniania " + (
        "(średnia ± std po foldach)" if stat == "mean" else "(mediana po oknach RMSE okna - odporna na pojedyncze okna)"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_profiles(prof, L, out_path, targets):
    p = prof[prof["L_cm"] == L]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    for ax, t in zip(axes, targets):
        unit = TARGET_UNITS[t]
        for (model, mode), g in p[p["target"] == t].groupby(["model", "mode"], sort=False):
            ax.plot(g["z_m"], g["rmse"], lw=1, ls="-" if mode == "local" else "--", label=f"{model} ({mode})")
        ax.set_ylabel(f"RMSE {'p' if t == 'pressure' else t} [{unit}]")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("położenie wzdłuż włókna z [m]")
    fig.suptitle(f"Błąd w poszczególnych oknach L = {L} cm (predykcje out-of-fold)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_sensitivity(sens, L, out_path):
    panels = [("ST_X", "dμ_X/dT [GHz/°C]"), ("ST_Y", "dμ_Y/dT [GHz/°C]"),
              ("C_X", "C_X [GHz/MPa]"), ("C_Y", "C_Y [GHz/MPa]"),
              ("ST_m", "dm/dT [GHz/°C]"), ("C_d", "C_d = C_X - C_Y [GHz/MPa]")]
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex=True)
    for ax, (col, lab) in zip(axes.ravel(), panels):
        ax.plot(sens["z_m"], sens[col], lw=1)
        se = f"se_{col}"
        if se in sens:
            ax.fill_between(sens["z_m"], sens[col] - 2 * sens[se], sens[col] + 2 * sens[se], alpha=0.25)
        ax.axhline(sens[col].median(), color="k", lw=0.7, ls="--")
        # odporna skala: pojedyncze okna brzegowe nie spłaszczają profilu; punkty poza skalą zaznaczone
        lo, hi = np.nanquantile(sens[col], [0.02, 0.98])
        pad = 0.5 * (hi - lo) or 1e-3
        ax.set_ylim(lo - pad, hi + pad)
        out = (sens[col] < lo - pad) | (sens[col] > hi + pad)
        if out.any():
            ax.scatter(sens["z_m"][out], np.clip(sens[col][out], lo - pad, hi + pad), marker="x", c="r", s=25,
                       label=f"poza skalą: {out.sum()}")
            ax.legend(fontsize=7, loc="best")
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("z [m]")
    fig.suptitle(f"Lokalne czułości (AN-JOINT na wszystkich rekordach), okna L = {L} cm; pasmo = ±2 SE")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- uruchomienie

def run_spatial(cfg, out_dir, df):
    df = df[df["is_joint_regression"] == True].reset_index(drop=True)
    cache = build_bin_cache(cfg, df)
    base = cfg["base_bin_cm"]
    targets = get_targets(cfg)
    t_col = targets[1]
    # modele liczą w temperaturze względem referencji (AN-JOINT nie ma wyrazu wolnego);
    # błędy (pred - true) i RMSE nie zależą od tego przesunięcia
    y = df[targets].to_numpy(float)
    y[:, 1] -= cfg["temperature_reference"]

    folds = select_folds(df, cfg)
    models, modes = cfg["models"], cfg["modes"]

    # kontrola zgodności: okno "full" musi odtworzyć mu_X, mu_Y z paired_features
    full = window_signals(cache, "full", base, 0)
    dev = max(np.nanmax(np.abs(full["X"][:, 0] - df["mu_X"])), np.nanmax(np.abs(full["Y"][:, 0] - df["mu_Y"])))
    print(f"Kontrola: max |mu(okno full) - mu(paired_features)| = {dev:.2e} GHz")

    summary, prof_rows, sens_tables = [], [], {}
    for L in cfg["window_lengths_cm"]:
        sig = window_signals(cache, L, base, cfg["min_coverage"])
        W = sig["n_windows"]
        L_m = (sig["bins_used"] / W) * base / 100
        print(f"\n>>> L = {L} cm: {W} okien, brak sygnału w {np.isnan(sig['X']).mean() * 100:.2f}% (okno, próbka)")
        for model in models:
            for mode in (modes if W > 1 else ["local"]):
                fold_rmse = []
                sse = {t: np.zeros(W) for t in targets}
                cnt = {t: np.zeros(W) for t in targets}
                for g, mask in folds:
                    tr, te = ~mask, mask
                    ep, et = _fold_errors(model, mode, sig["X"], sig["Y"], y[tr], tr, te, y[te])
                    fold_rmse.append((_rmse(ep), _rmse(et), float(np.mean(np.isfinite(ep)))))
                    for t, e in zip(targets, (ep, et)):
                        sse[t] += np.nansum(e ** 2, axis=0)
                        cnt[t] += np.isfinite(e).sum(axis=0)
                fr = np.array(fold_rmse)
                with np.errstate(invalid="ignore"):
                    w_rmse = {t: np.sqrt(sse[t] / cnt[t]) for t in targets}
                summary.append({
                    "L_label": str(L), "L_cm": L_m * 100, "L_m": L_m, "n_windows": W, "model": model,
                    "mode": mode if W > 1 else "local=global",
                    "pressure_rmse": fr[:, 0].mean(), "pressure_rmse_std": fr[:, 0].std(ddof=1),
                    f"{t_col}_rmse": fr[:, 1].mean(), f"{t_col}_rmse_std": fr[:, 1].std(ddof=1),
                    # odporna miara: mediana po oknach RMSE okna (wszystkie foldy razem)
                    "pressure_rmse_wmedian": np.nanmedian(w_rmse["pressure"]),
                    f"{t_col}_rmse_wmedian": np.nanmedian(w_rmse[t_col]),
                    "coverage": fr[:, 2].mean(),
                })
                print(f"  {model:<9} {summary[-1]['mode']:<13} p_RMSE={summary[-1]['pressure_rmse']:.3f}"
                      f"  {t_col}_RMSE={summary[-1][f'{t_col}_rmse']:.3f}")
                if L in cfg["profile_lengths_cm"]:
                    for t in targets:
                        r = w_rmse[t]
                        for w in range(W):
                            prof_rows.append({"L_cm": L, "model": model, "mode": mode, "target": t,
                                              "window": w, "z_m": sig["z"][w], "rmse": r[w]})
        if L in cfg["sensitivity_lengths_cm"]:
            sens_tables[L] = _sensitivities(sig, y)

    table = pd.DataFrame(summary)
    # "full" jest wspólne dla obu trybów - powielamy, żeby krzywe local i global kończyły się w tym samym punkcie
    full_rows = table[table["L_label"] == "full"]
    table = pd.concat([table[table["L_label"] != "full"]] +
                      [full_rows.assign(mode=m) for m in modes], ignore_index=True)
    write_table(table, out_dir, "rmse_vs_L", "RMSE w funkcji długości okna (średnia i std po foldach)")
    write_table(_min_length(table, cfg["targets"], cfg["thresholds"], targets), out_dir, "min_window_length",
                "Minimalna długość okna spełniająca kryterium (L i wszystkie dłuższe okna)", digits=1)
    _plot_rmse_vs_L(table, os.path.join(out_dir, "rmse_vs_L.png"), targets)
    _plot_rmse_vs_L(table, os.path.join(out_dir, "rmse_vs_L_window_median.png"), targets, stat="median")

    if prof_rows:
        prof = pd.DataFrame(prof_rows)
        prof.to_csv(os.path.join(out_dir, "window_rmse_profiles.csv"), index=False)
        bad = _outlier_windows(prof)
        write_table(bad if not bad.empty else pd.DataFrame({"info": ["brak okien odstających"]}),
                    out_dir, "outlier_windows", "Okna z RMSE > mediana + 3 MAD (systematycznie większy błąd)")
        for L in cfg["profile_lengths_cm"]:
            _plot_profiles(prof, L, os.path.join(out_dir, f"window_rmse_L{L}cm.png"), targets)

    hom_rows = []
    for L, sens in sens_tables.items():
        sens.to_csv(os.path.join(out_dir, f"sensitivity_L{L}cm.csv"), index=False)
        _plot_sensitivity(sens, L, os.path.join(out_dir, f"sensitivity_L{L}cm.png"))
        hom_rows.append(_homogeneity(sens).assign(L_cm=L))
    if hom_rows:
        hom = pd.concat(hom_rows, ignore_index=True)
        write_table(hom[["L_cm"] + [c for c in hom.columns if c != "L_cm"]], out_dir, "sensitivity_homogeneity",
                    "Jednorodność lokalnych czułości (std/SE >> 1: niejednorodność ponad niepewność dopasowania)",
                    digits=4)

    print("\n" + table[["L_label", "n_windows", "model", "mode"] + [f"{t}_rmse" for t in targets]].to_string(index=False))
    return {"table": table}
