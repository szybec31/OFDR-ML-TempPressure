"""
Analiza powtarzalności i dryftu zero_end (zadanie 9 z listy prowadzącego).

Pomiary zero_end nie biorą udziału w uczeniu żadnego modelu: modele ML uczą się na
is_joint_regression (bez zero_end), modele fizyczne kalibrują się na tych samych rekordach,
a przelicznik dryftu na T i p (AN-JOINT) jest dopasowywany bez zero_end.
Wyjątek: ablacja 4 (include_zero_end_train), która celowo pokazuje skutek błędu.

Budowa serii (z czasów akwizycji w nagłówkach plików OBR): seria to JEDEN przebieg ciśnienia
trwający 35-55 min: 10 szybkich pomiarów zero_start (co ~6 s), kolejne poziomy ciśnienia
(po 10 pomiarów), na końcu 10 pomiarów zero_end. "Powtórzenie" (rep_idx) to numer pomiaru
w bloku, nie osobny cykl. Dryft = zmiana sygnału przy 0 MPa między początkiem a końcem serii.

Parametry każdej serii (series_parameters):
    n par, dT/Tp, czas zero_start -> zero_end, kierunek przebiegu ciśnienia (rosnący / malejący),
    ciśnienie bezpośrednio przed zero_end, ciśnienie maksymalne,
    ΔX0 = X(zero_end) - X(zero_start), ΔY0 = Y(zero_end) - Y(zero_start),
    Δm0 = (ΔX0 + ΔY0)/2 (wspólna), Δd0 = ΔX0 - ΔY0 (różnicowa) [GHz] - średnia i std po parach
    (kolumny dX0, dY0, dm0, dd0; X, Y = średnie przesunięcie w ROI kanału X, Y),
    ΔT_drift, Δp_drift - dryft przeliczony na temperaturę i ciśnienie modelem fizycznym (AN-JOINT),
    tempo dryftu [°C/h, MPa/h], szum krótkoterminowy (std w bloku zero_start).

Odpowiedzi na pytania (answers):
    1. składowa wspólna vs różnicowa,
    2. zależność od temperatury (Spearman po seriach),
    3. zależność od ciśnienia maksymalnego / historii ciśnienia,
    4. wpływ na estymację T vs p (dryft względem RMSE modelu),
    5. wskaźnik jakości: Q = max(|ΔT_drift| / RMSE_T, |Δp_drift| / RMSE_p) i jego weryfikacja:
       czy liniowa w czasie korekta dryftu zmniejsza błędy pomiarów z tej serii (out-of-fold).
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from baselines.models.physical_model import JointPhysicalModel


# ---------------------------------------------------------------- czasy akwizycji

def _acquired_at(path):
    """Pierwsza linia pliku OBR: 'Acquired on 9/11/2025 at 12:28:54'."""
    try:
        with open(path, "r", errors="ignore") as f:
            line = f.readline().strip()
        parts = line.replace("Acquired on", "").split("at")
        return datetime.strptime(parts[0].strip() + " " + parts[1].strip(), "%m/%d/%Y %H:%M:%S")
    except Exception:
        return pd.NaT


def acquisition_times(df, cache_path):
    """sample_id -> czas akwizycji (z nagłówka pliku kanału Y); cache w CSV."""
    ids = pd.Series(df["sample_id"].unique())
    if os.path.exists(cache_path):
        c = pd.read_csv(cache_path, parse_dates=["acquired_at"])
        if set(ids) <= set(c["sample_id"]):
            return c.set_index("sample_id")["acquired_at"]
    c = pd.DataFrame({"sample_id": ids, "acquired_at": [_acquired_at(p) for p in ids]})
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    c.to_csv(cache_path, index=False)
    return c.set_index("sample_id")["acquired_at"]


# ---------------------------------------------------------------- przebieg serii

def series_sequence(df, times):
    """Dla każdej serii: początek, koniec, kierunek przebiegu ciśnienia, ciśnienie przed zero_end."""
    d = df.assign(t=df["sample_id"].map(times))
    rows = []
    for sid, g in d.groupby("series_id"):
        g = g.sort_values("t")
        zs, ze, pp = (g[g["point_type"] == k] for k in ("zero_start", "zero_end", "pressure_point"))
        t0 = zs["t"].min()
        t1 = ze["t"].min() if len(ze) else pd.NaT
        rho = spearmanr(pp["t"].astype("int64"), pp["pressure"]).correlation if len(pp) > 2 else np.nan
        before = pp[pp["t"] < t1]["pressure"].iloc[-1] if len(ze) and len(pp[pp["t"] < t1]) else np.nan
        rows.append({
            "series_id": sid, "t_start": t0, "t_zero_end": t1,
            "duration_min": (t1 - t0).total_seconds() / 60 if pd.notna(t1) else np.nan,
            "sweep": "rosnący" if rho > 0.5 else ("malejący" if rho < -0.5 else "mieszany"),
            "p_before_zero_end": before, "p_max": g["pressure"].max(),
            "zero_start_is_first": bool(len(zs) and zs["t"].min() <= g["t"].min()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- dryft na parach

def drift_pairs(df, t_col, t_ref):
    """
    Pary zero_end <-> zero_start (ten sam numer pomiaru w bloku, ta sama seria): Δ sygnału
    oraz dryft przeliczony na T i p modelem fizycznym dopasowanym BEZ zero_end.
    """
    joint = df[df["is_joint_regression"] == True]
    phys = JointPhysicalModel(t_ref=t_ref).fit(joint[["mu_X", "mu_Y"]], joint[["pressure", t_col]])
    start = df[df["point_type"] == "zero_start"].set_index("sample_id")
    end = df[(df["point_type"] == "zero_end") & df["ref"].isin(start.index)]
    if end.empty:
        return pd.DataFrame(), phys
    s = start.loc[end["ref"]]
    pe = phys.predict(end[["mu_X", "mu_Y"]].to_numpy())
    ps = phys.predict(s[["mu_X", "mu_Y"]].to_numpy())
    dx = end["mu_X"].to_numpy() - s["mu_X"].to_numpy()
    dy = end["mu_Y"].to_numpy() - s["mu_Y"].to_numpy()
    out = pd.DataFrame({
        "sample_id": end["sample_id"].to_numpy(), "series_id": end["series_id"].to_numpy(),
        t_col: end[t_col].to_numpy(), "rep_idx": end["rep_idx"].to_numpy(),
        "delta_mu_X": dx, "delta_mu_Y": dy, "delta_m": (dx + dy) / 2, "delta_d": dx - dy,
        "dT_drift": pe[:, 1] - ps[:, 1], "dp_drift": pe[:, 0] - ps[:, 0],
    })
    return out, phys


def series_parameters(df, pairs, seq, t_col):
    """Parametry dryftu dla każdej serii (średnia i odchylenie po parach)."""
    zs = df[df["point_type"] == "zero_start"]
    noise = zs.groupby("series_id").agg(noise_m_GHz=("m", "std"), noise_d_GHz=("d", "std"))
    agg = pairs.groupby("series_id").agg(
        n_pairs=("delta_m", "size"), **{t_col: (t_col, "first")},
        # parametry z maila prowadzącego: ΔX0, ΔY0, Δm0 = (ΔX0 + ΔY0)/2, Δd0 = ΔX0 - ΔY0
        dX0=("delta_mu_X", "mean"), dX0_std=("delta_mu_X", "std"),
        dY0=("delta_mu_Y", "mean"), dY0_std=("delta_mu_Y", "std"),
        dm0=("delta_m", "mean"), dm0_std=("delta_m", "std"),
        dd0=("delta_d", "mean"), dd0_std=("delta_d", "std"),
        dT_drift=("dT_drift", "mean"), dT_drift_std=("dT_drift", "std"),
        dp_drift=("dp_drift", "mean"), dp_drift_std=("dp_drift", "std"),
    )
    out = seq.set_index("series_id").join(agg, how="inner").join(noise, how="left")
    hours = out["duration_min"] / 60
    out["dT_rate_per_h"] = out["dT_drift"] / hours
    out["dp_rate_per_h"] = out["dp_drift"] / hours
    out["abs_dm0_over_abs_dd0"] = out["dm0"].abs() / out["dd0"].abs()
    return out.reset_index().sort_values(t_col)


# ---------------------------------------------------------------- korekta dryftu (weryfikacja wskaźnika)

def drift_correction_check(preds, params, times, targets, models):
    """
    Zakładamy liniowy w czasie dryft od zero_start do zero_end. Przewidywany błąd pomiaru z chwili t:
        offset(t) = drift * (t - t_start) / (t_zero_end - t_start).
    Jeśli dryft jest rzeczywistym zjawiskiem czasowym, odjęcie offsetu od predykcji out-of-fold
    zmniejsza RMSE w tej serii. Zwraca RMSE przed/po dla każdej serii i modelu.
    """
    p_col, t_col = targets
    test = preds[(preds["subset"] == "test") & preds["model"].isin(models)].copy()
    test["t"] = test["sample_id"].map(times)
    rows = []
    for _, s in params.iterrows():
        part = test[test["series_id"] == s["series_id"]]
        if part.empty or not np.isfinite(s["duration_min"]):
            continue
        frac = ((part["t"] - s["t_start"]).dt.total_seconds() / 60 / s["duration_min"]).clip(0, 1)
        for m, g in part.groupby("model"):
            f = frac.loc[g.index]
            et = g[f"{t_col}_pred"] - g[t_col]
            ep = g[f"{p_col}_pred"] - g[p_col]
            et_c = et - s["dT_drift"] * f
            ep_c = ep - s["dp_drift"] * f
            rows.append({
                "series_id": s["series_id"], t_col: s[t_col], "model": m, "n": len(g),
                "rmse_T_before": np.sqrt(np.mean(et ** 2)), "rmse_T_after": np.sqrt(np.mean(et_c ** 2)),
                "rmse_p_before": np.sqrt(np.mean(ep ** 2)), "rmse_p_after": np.sqrt(np.mean(ep_c ** 2)),
                # zgodność znaku/kształtu: korelacja błędu z przewidywanym offsetem w czasie
                "corr_errT_offset": np.corrcoef(et, f)[0, 1] * np.sign(s["dT_drift"]) if f.std() > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def series_error_profile(preds, params, times, targets, model):
    """
    Przebieg błędu out-of-fold modelu w czasie serii: zero_start, pierwszy i ostatni poziom ciśnienia,
    zero_end oraz nachylenie błędu pomiarów ciśnieniowych w czasie. Pozwala odróżnić dryft
    narastający w czasie od skoku przy obciążeniu / odciążeniu (histerezy).
    """
    p_col, t_col = targets
    d = preds[(preds["model"] == model)].copy()
    d["t"] = d["sample_id"].map(times)
    rows = []
    for _, s in params.iterrows():
        g = d[d["series_id"] == s["series_id"]]
        if g.empty:
            continue
        g = g.assign(min=(g["t"] - s["t_start"]).dt.total_seconds() / 60,
                     eT=g[f"{t_col}_pred"] - g[t_col], ep=g[f"{p_col}_pred"] - g[p_col])
        pp = g[g["point_type"] == "pressure_point"]
        lv = pp.groupby("pressure").agg(min=("min", "mean"), eT=("eT", "mean")).sort_values("min")
        slope = np.polyfit(pp["min"] / 60, pp["eT"], 1)[0] if len(pp) > 2 else np.nan
        e = lambda k: g.loc[g["point_type"] == k, "eT"].mean()
        rows.append({
            "series_id": s["series_id"], t_col: s[t_col], "sweep": s["sweep"],
            "errT_zero_start": e("zero_start"), "errT_first_level": lv["eT"].iloc[0] if len(lv) else np.nan,
            "errT_last_level": lv["eT"].iloc[-1] if len(lv) else np.nan, "errT_zero_end": e("zero_end"),
            "jump_start": (lv["eT"].iloc[0] - e("zero_start")) if len(lv) else np.nan,
            "jump_end": (e("zero_end") - lv["eT"].iloc[-1]) if len(lv) else np.nan,
            "slope_points_per_h": slope, "drift_rate_per_h": s["dT_rate_per_h"],
            "rmse_T_series": np.sqrt(np.mean(pp["eT"] ** 2)), "rmse_p_series": np.sqrt(np.mean(pp["ep"] ** 2)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- odpowiedzi

def _spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan, np.nan
    r = spearmanr(x[ok], y[ok])
    return r.correlation, r.pvalue


def answers(params, pairs, rmse_ref, corr_check, t_col, rd_summary=None, profile=None):
    """Tabela odpowiedzi na 5 pytań; rmse_ref = {'T': ..., 'p': ...} modelu odniesienia."""
    rows = []
    am, ad = pairs["delta_m"].abs().mean(), pairs["delta_d"].abs().mean()
    aT, ap = pairs["dT_drift"].abs().mean(), pairs["dp_drift"].abs().mean()
    rows.append({"pytanie": "1. wspólna czy różnicowa",
                 "wynik": f"|Δm| = {am:.3f} GHz, |Δd| = {ad:.3f} GHz (stosunek {am / ad:.1f}); "
                          f"po przeliczeniu: |ΔT| = {aT:.3f} °C, |Δp| = {ap:.3f} MPa",
                 "wniosek": "dryft głównie w składowej wspólnej (temperaturowej)" if am > 2 * ad
                            else "dryft w obu składowych porównywalny"})

    r, pv = _spearman(params[t_col].to_numpy(float), params["dT_drift"].to_numpy(float))
    ra, pva = _spearman(params[t_col].to_numpy(float), params["dT_drift"].abs().to_numpy(float))
    rows.append({"pytanie": f"2. zależność od temperatury ({t_col})",
                 "wynik": f"Spearman(T, ΔT_drift) = {r:.2f} (p = {pv:.2f}); "
                          f"Spearman(T, |ΔT_drift|) = {ra:.2f} (p = {pva:.2f}); n = {len(params)} serii",
                 "wniosek": "brak istotnej zależności (mała liczba serii)" if not (pv < 0.05 or pva < 0.05)
                            else "zależność istotna na poziomie 0,05"})

    pmax = params["p_max"].nunique()
    grp = params.groupby("sweep")["dT_drift"].agg(["mean", "count"])
    rd, pvd = _spearman(params["duration_min"].to_numpy(float), params["dT_drift"].abs().to_numpy(float))
    rows.append({"pytanie": "3. zależność od maksymalnego ciśnienia / historii ciśnienia",
                 "wynik": (f"p_max we wszystkich seriach = {params['p_max'].iloc[0]:g} MPa (nie da się testować); "
                           if pmax == 1 else f"p_max: {sorted(params['p_max'].unique())}; ")
                          + "; ".join(f"przebieg {k}: śr. ΔT_drift = {v['mean']:+.3f} °C (n = {int(v['count'])})"
                                      for k, v in grp.iterrows())
                          + f"; Spearman(czas serii, |ΔT_drift|) = {rd:.2f} (p = {pvd:.2f})",
                 "wniosek": "maksymalne ciśnienie stałe - pytanie nierozstrzygalne na tych danych; "
                            "porównanie kierunku przebiegu i czasu trwania serii w tabeli"})

    rows.append({"pytanie": "4. wpływ na T czy p",
                 "wynik": f"|ΔT_drift| / RMSE_T = {aT / rmse_ref['T']:.2f}, "
                          f"|Δp_drift| / RMSE_p = {ap / rmse_ref['p']:.2f} "
                          f"(RMSE modelu odniesienia: {rmse_ref['T']:.3f} °C, {rmse_ref['p']:.3f} MPa)"
                          + ("" if rd_summary is None else
                             f"; zero_end out-of-fold: średni |błąd T| = {rd_summary['abs_err_T_mean']:.3f} °C, "
                             f"|błąd p| = {rd_summary['abs_err_pressure_mean']:.3f} MPa"),
                 "wniosek": "dryft wpływa głównie na estymację temperatury"
                            if aT / rmse_ref["T"] > 2 * ap / rmse_ref["p"] else
                            ("głównie na ciśnienie" if ap / rmse_ref["p"] > 2 * aT / rmse_ref["T"]
                             else "porównywalnie na T i p")})

    if corr_check is not None and not corr_check.empty:
        chg_T = corr_check["rmse_T_after"].mean() / corr_check["rmse_T_before"].mean() - 1
        chg_p = corr_check["rmse_p_after"].mean() / corr_check["rmse_p_before"].mean() - 1
        frac_better = (corr_check["rmse_T_after"] < corr_check["rmse_T_before"]).mean()
        wyn = (f"liniowa korekta dryftu w czasie zmienia RMSE_T o {chg_T * 100:+.1f} % i RMSE_p o "
               f"{chg_p * 100:+.1f} % (średnio po seriach i modelach; ujemne = poprawa); RMSE_T maleje tylko w "
               f"{frac_better * 100:.0f} % par (seria, model)")
        r = pv = np.nan
        if profile is not None and len(profile) >= 4:
            q = profile[t_col].map(params.set_index(t_col)["dT_drift"]).abs() / rmse_ref["T"]
            r, pv = _spearman(q.to_numpy(float), profile["rmse_T_series"].to_numpy(float))
            wyn += f"; Spearman(Q_T, RMSE_T pomiarów serii) = {r:.2f} (p = {pv:.2f})"
        predictive = np.isfinite(pv) and pv < 0.05 and r > 0
        if chg_T < -0.02:
            concl = "dryft z zero_end przewiduje część błędu pomiarów serii - nadaje się do korekty predykcji"
        elif predictive:
            concl = ("korekta liniowa nie działa, ale Q rośnie razem z błędem serii - nadaje się na ostrzeżenie")
        else:
            concl = ("dryft nie narasta liniowo w czasie (często skok przy odciążeniu) i Q nie przewiduje błędu "
                     "pomiarów serii: zero_end mówi o stabilności punktu zerowego, ale nie nadaje się ani do "
                     "korekty, ani jako wiarygodne ostrzeżenie dla pozostałych pomiarów serii")
    else:
        wyn, concl = "brak predykcji out-of-fold dla serii z zero_end", "-"
    rows.append({"pytanie": "5. wskaźnik jakości / ostrzeżenie",
                 "wynik": "Q = max(|ΔT_drift| / RMSE_T, |Δp_drift| / RMSE_p); Q > 1: ostrzeżenie, Q > 2: "
                          "predykcja niewiarygodna. " + wyn,
                 "wniosek": concl})
    return pd.DataFrame(rows)


def quality_flags(params, rmse_ref):
    q_t = params["dT_drift"].abs() / rmse_ref["T"]
    q_p = params["dp_drift"].abs() / rmse_ref["p"]
    q = np.maximum(q_t, q_p)
    return params.assign(Q_T=q_t, Q_p=q_p, Q=q,
                         flag=np.where(q > 2, "niewiarygodna", np.where(q > 1, "ostrzeżenie", "ok")))


# ---------------------------------------------------------------- wykres

def plot_drift(params, pairs, t_col, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    ax.scatter(pairs["delta_m"], pairs["delta_d"], s=14, c=pairs[t_col], cmap="viridis")
    lim = max(pairs["delta_m"].abs().max(), pairs["delta_d"].abs().max()) * 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Δm (wspólna) [GHz]"); ax.set_ylabel("Δd (różnicowa) [GHz]")
    ax.set_title("Dryft par zero_end - zero_start")
    ax = axes[1]
    for sweep, g in params.groupby("sweep"):
        ax.errorbar(g[t_col], g["dT_drift"], yerr=g["dT_drift_std"], fmt="o", capsize=3, label=f"przebieg {sweep}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(f"{t_col} [°C]"); ax.set_ylabel("ΔT_drift [°C]"); ax.set_title("Dryft temperatury wg serii")
    ax.legend(fontsize=8)
    ax = axes[2]
    ax.errorbar(params[t_col], params["dp_drift"], yerr=params["dp_drift_std"], fmt="s", capsize=3, color="C3")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(f"{t_col} [°C]"); ax.set_ylabel("Δp_drift [MPa]"); ax.set_title("Dryft ciśnienia wg serii")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
