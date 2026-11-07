"""
Punkt 11: końcowe porównanie modeli - jedna tabela z wyników kilku wariantów jednego uruchomienia.

Klucz konfiguracji "final_table" (nazwy wariantów z listy "variants"):
    {"main": "condition",          # schemat główny: RMSE / MAE T, MSE / MAE p, max |błąd|, bias
     "loso": "series", "loto": "temperature", "lopo": "pressure",
     "conformal": "condition",     # wariant, z którego biorą się szerokości przedziałów predykcji
     "zero_end": "condition",      # wariant, z którego bierze się błąd na zero_end (out-of-fold)
     "reduced_calibration": null,  # opcjonalnie: ścieżka do min_states.csv z run_reduced (empiryczne minimum)
     "reduced_threshold": 10}      # próg pogorszenia [%] dla empirycznej minimalnej liczby stanów

Konwencje:
- RMSE, MAE, MSE, bias - średnia po foldach referencyjnych danego wariantu (jak results_table),
- max |błąd| - maksimum po wszystkich predykcjach testowych foldów referencyjnych (pooled),
- schematy leave-one-*-out: RMSE p i T (średnia po foldach) i liczba foldów, w których model działał,
- liczba parametrów: mediana po foldach schematu głównego (count_parameters, bez standaryzacji);
  "nonparametric" = liczba rośnie z danymi (SVR, GP, RF),
- minimalna liczba stanów: z rejestru modeli (identyfikowalność) i - jeżeli podano wynik run_reduced -
  empirycznie (najmniejsza liczba stanów z pogorszeniem RMSE p i T <= próg, najlepsza metoda wyboru),
- szerokość przedziału predykcji: 2q z grupowej predykcji konforemnej (90% i 95%), z pokryciem
  leave-one-block-out,
- zero_end: średni |błąd| T i p oraz Residual Drift na zero_end (predykcje out-of-fold).
"""
import numpy as np
import pandas as pd

from .conformal import conformal_wide
from .models import get_model_spec

FINAL_DEFAULTS = {"main": "condition", "loso": "series", "loto": "temperature", "lopo": "pressure",
                  "conformal": "condition", "zero_end": "condition",
                  "reduced_calibration": None, "reduced_threshold": 10}


def _empirical_min_states(path, threshold):
    if not path:
        return {}
    ms = pd.read_csv(path)
    col = f"n_both_{int(threshold)}pct"
    if col not in ms:
        raise ValueError(f"final_table.reduced_calibration: brak kolumny {col} w {path}")
    best = ms.groupby("model")[col].min()
    return best.to_dict()


def build_final_table(infos, cfg, targets):
    ft = dict(FINAL_DEFAULTS)
    ft.update(cfg.get("final_table") or {})
    p, t = targets
    missing = [ft[k] for k in ("main", "loso", "loto", "lopo", "conformal", "zero_end") if ft[k] not in infos]
    if missing:
        print(f"UWAGA (tabela końcowa): brak wariantów {sorted(set(missing))} - odpowiednie kolumny będą puste")

    models = list(cfg["models"])
    table = pd.DataFrame({"model": models})

    main = infos.get(ft["main"])
    if main is not None:
        s = main["summary_common"].set_index("model")
        pooled = main["pooled"].set_index("model") if not main["pooled"].empty else pd.DataFrame()
        cols = {f"{t}_rmse_mean": f"RMSE_{t}", f"{t}_mae_mean": f"MAE_{t}",
                f"{p}_mse_mean": f"MSE_{p}", f"{p}_rmse_mean": f"RMSE_{p}", f"{p}_mae_mean": f"MAE_{p}",
                f"{t}_bias_mean": f"bias_{t}", f"{p}_bias_mean": f"bias_{p}", "n_folds": "n_folds_main"}
        for src, dst in cols.items():
            table[dst] = table["model"].map(s[src]) if src in s else np.nan
        for tt in (t, p):
            col = f"{tt}_maxae"
            table[f"maxAE_{tt}"] = table["model"].map(pooled[col]) if col in pooled else np.nan
        unc = main.get("uncertainty")
        if unc is not None:
            # pkt 10 w schemacie głównym: przedział RMSE i różnicy RMSE względem modelu analitycznego
            us = unc["summary"].set_index("model")
            for c in [f"{tt}_rmse_{q}" for tt in (p, t) for q in ("lo", "hi")] + \
                     [f"{tt}_rmse_diff_vs_ref_{q}" for tt in (p, t) for q in ("median", "lo", "hi")]:
                if c in us:
                    table[f"main_{c}"] = table["model"].map(us[c])

    for key, name in [("loso", "LOSO"), ("loto", "LOTO"), ("lopo", "LOPO")]:
        info = infos.get(ft[key])
        if info is None:
            continue
        s = info["summary_common"].set_index("model")
        for tt in (p, t):
            table[f"{name}_RMSE_{tt}"] = table["model"].map(s[f"{tt}_rmse_mean"])
        n_all = len(info["common_folds"])
        table[f"{name}_n_folds"] = table["model"].map(lambda m, s=s, n=n_all: (
            f"{int(s.loc[m, 'n_folds'])}/{n}" if m in s.index else f"0/{n}"))
        unc = info.get("uncertainty")
        if unc is not None:
            us = unc["summary"].set_index("model")
            for tt in (p, t):
                for q in ("lo", "hi"):
                    c = f"{tt}_rmse_{q}"
                    if c in us:
                        table[f"{name}_RMSE_{tt}_{q}"] = table["model"].map(us[c])

    if main is not None and "n_params" in main["folds"]:
        f = main["folds"][main["folds"]["status"] == "ok"]
        table["n_params"] = table["model"].map(f.groupby("model")["n_params"].median())
        table["param_kind"] = table["model"].map(f.groupby("model")["param_kind"].first())
    table["min_states_identifiable"] = [get_model_spec(m).min_states for m in models]
    emp = _empirical_min_states(ft["reduced_calibration"], ft["reduced_threshold"])
    if emp:
        table[f"min_states_empirical_{int(ft['reduced_threshold'])}pct"] = table["model"].map(emp)

    conf = infos.get(ft["conformal"])
    if conf is not None and not conf["conformal"].empty:
        w = conformal_wide(conf["conformal"], targets).set_index("model")
        for c in w.columns:
            table[c] = table["model"].map(w[c])

    ze = infos.get(ft["zero_end"])
    if ze is not None and not ze["rd_summary"].empty:
        rd = ze["rd_summary"].set_index("model")
        table[f"zero_end_absErr_{t}"] = table["model"].map(rd["abs_err_T_mean"])
        table[f"zero_end_absErr_{p}"] = table["model"].map(rd["abs_err_pressure_mean"])
        table["zero_end_RD"] = table["model"].map(rd["RD_mean"])
    return table, ft


def compact_final_table(table, targets, ft):
    """Skrócona wersja do sprawozdania: najważniejsze kolumny z jednostkami."""
    p, t = targets

    def fmt(v, d=3):
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (
            "inf" if isinstance(v, float) and np.isinf(v) else f"{v:.{d}f}")

    rows = []
    for _, r in table.iterrows():
        row = {"model": r["model"]}
        for c, name in [(f"RMSE_{t}", f"RMSE {t} [°C]"), (f"MAE_{t}", f"MAE {t} [°C]"),
                        (f"MSE_{p}", "MSE p [MPa^2]"), (f"MAE_{p}", "MAE p [MPa]"),
                        (f"maxAE_{t}", f"max|e| {t} [°C]"), (f"maxAE_{p}", "max|e| p [MPa]"),
                        (f"bias_{t}", f"bias {t} [°C]"), (f"bias_{p}", "bias p [MPa]")]:
            row[name] = fmt(r.get(c, np.nan), 4 if c.startswith("MSE") else 3)
        if f"main_{t}_rmse_lo" in r:
            row[f"RMSE p / {t}: przedział (replikacje)"] = (
                f"[{fmt(r[f'main_{p}_rmse_lo'])}; {fmt(r[f'main_{p}_rmse_hi'])}] / "
                f"[{fmt(r[f'main_{t}_rmse_lo'])}; {fmt(r[f'main_{t}_rmse_hi'])}]")
        if f"main_{t}_rmse_diff_vs_ref_median" in r and not pd.isna(r[f"main_{t}_rmse_diff_vs_ref_median"]):
            row[f"dRMSE vs analityczny p / {t}"] = (
                f"{fmt(r[f'main_{p}_rmse_diff_vs_ref_median'])} [{fmt(r[f'main_{p}_rmse_diff_vs_ref_lo'])}; "
                f"{fmt(r[f'main_{p}_rmse_diff_vs_ref_hi'])}] / {fmt(r[f'main_{t}_rmse_diff_vs_ref_median'])} "
                f"[{fmt(r[f'main_{t}_rmse_diff_vs_ref_lo'])}; {fmt(r[f'main_{t}_rmse_diff_vs_ref_hi'])}]")
        for name in ("LOSO", "LOTO", "LOPO"):
            if f"{name}_RMSE_{p}" in r:
                row[f"{name} RMSE p / {t}"] = (f"{fmt(r[f'{name}_RMSE_{p}'])} / {fmt(r[f'{name}_RMSE_{t}'])}"
                                               f" ({r[f'{name}_n_folds']})")
        if "n_params" in r:
            row["l. parametrów"] = "" if pd.isna(r["n_params"]) else (
                f"{int(r['n_params'])}" + (" (nieparam.)" if r.get("param_kind") == "nonparametric" else ""))
        emp = f"min_states_empirical_{int(ft['reduced_threshold'])}pct"
        row["min. l. stanów"] = (f"{int(r['min_states_identifiable'])}"
                                 + (f" / {fmt(r[emp], 0)}" if emp in r and not pd.isna(r[emp]) else ""))
        for lvl in (90, 95):
            if f"{p}_PI{lvl}_width" in r:
                row[f"PI{lvl} p / {t}"] = f"±{fmt(r[f'{p}_PI{lvl}_width'] / 2)} / ±{fmt(r[f'{t}_PI{lvl}_width'] / 2)}"
        if "zero_end_RD" in r:
            row[f"zero_end |e| p / {t}"] = f"{fmt(r[f'zero_end_absErr_{p}'])} / {fmt(r[f'zero_end_absErr_{t}'])}"
        rows.append(row)
    return pd.DataFrame(rows)
