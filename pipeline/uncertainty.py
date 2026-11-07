"""
Punkt 10: niepewność wyników przez replikacje z ponownym dopasowaniem modeli.

Replikacja b (bootstrap blokowy, dane zależne -> losowane bloki, nie pojedyncze obserwacje):
    1. w każdym foldzie zewnętrznym dane treningowe są losowane ze zwracaniem blokami
       (domyślnie blok = warunek (T, p), czyli 10 powtórzeń jednego stanu razem; "series_id" = całe
       serie), wszystkie modele są dopasowywane ponownie, a test folda pozostaje niezmieniony,
    2. (resample_test_folds) foldy testowe są losowane ze zwracaniem - wagi foldów w średniej;
       ten sam los dla wszystkich modeli, więc różnice między modelami są parowane,
    3. dodatkowo (full_data_fit) model jest dopasowywany do bootstrapowej próby CAŁEGO zbioru -
       z tych dopasowań pochodzą przedziały współczynników (np. modelu common/differential CD-LIN).
Zapisywane dla każdej replikacji: błędy (metryki per fold i zagregowane), współczynniki modeli,
różnice RMSE każdego modelu względem modelu analitycznego (odniesienia).

Hiperparametry modeli ML (tune):
    "fixed"         - wybrane w zwykłej walidacji dla danego folda (pełne dane folda), w replikacjach
                      tylko ponowne dopasowanie; szybkie, ale pomija niepewność wyboru hiperparametrów,
    "per_replicate" - pełne strojenie (GridSearchCV) w każdej replikacji; dokładniejsze, wolne.

Raport: mediana i przedział (domyślnie 95%, percentylowy) po replikacjach dla RMSE T, RMSE p,
różnicy RMSE względem modelu analitycznego (z odsetkiem replikacji, w których model jest lepszy)
oraz współczynników modeli.

Klucz konfiguracji "uncertainty" (n_replicates = 0 -> wyłączone):
    {"n_replicates": 200, "block": "condition_id", "resample_test_folds": true, "tune": "fixed",
     "reference": null, "ci": 0.95, "full_data_fit": true, "models": null, "n_jobs": -1}
"""
import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from baselines.models.physical_model import NotIdentifiableError
from .cv import _fit_ml
from .metrics import TARGET_UNITS, evaluate, get_targets
from .models import build_estimator, get_model_spec, model_coefficients
from .report import coefficient_summary, write_table

UNCERTAINTY_DEFAULTS = {
    "n_replicates": 0,
    "block": "condition_id",       # kolumna bloków losowanych w treningu: condition_id | series_id | sample_id
    "resample_test_folds": True,
    "tune": "fixed",               # fixed | per_replicate
    "reference": None,             # model analityczny odniesienia; None -> baseline (AN-*) albo pierwszy AN-*
    "ci": 0.95,
    "full_data_fit": True,
    "models": None,                # None = wszystkie modele z konfiguracji
    "n_jobs": -1,
}


def uncertainty_config(cfg):
    u = dict(UNCERTAINTY_DEFAULTS)
    u.update(cfg.get("uncertainty") or {})
    if u["tune"] not in ("fixed", "per_replicate"):
        raise ValueError(f"uncertainty.tune = {u['tune']!r}; dozwolone: fixed, per_replicate")
    return u


def reference_model(u, cfg, models):
    if u["reference"]:
        if u["reference"] not in models:
            raise ValueError(f"uncertainty.reference = {u['reference']!r} nie występuje w modelach {models}")
        return u["reference"]
    base = cfg.get("baseline")
    if base and base.startswith("AN-") and base in models:
        return base
    return next((m for m in models if m.startswith("AN-")), base)


def _block_resample(train, block, rng):
    """Bootstrap blokowy: len(bloków) bloków losowanych ze zwracaniem; kopie zachowują etykietę bloku."""
    groups = train.groupby(block, sort=False).indices
    keys = list(groups)
    draw = rng.integers(0, len(keys), size=len(keys))
    pos = np.concatenate([groups[keys[i]] for i in draw])
    return train.iloc[pos].reset_index(drop=True)


def _fit_one(spec, train_b, cfg, u, fixed_params, seed):
    targets = get_targets(cfg)
    if spec.kind == "physical":
        return build_estimator(spec, seed, cfg).fit(train_b, train_b[targets]), ["mu_X", "mu_Y"]
    feats = spec.features or cfg["features"]
    is_joint = train_b["is_joint_regression"] == True
    if cfg.get("include_zero_end_train", False):
        is_joint |= train_b["is_repeatability_test"] == True
    ml = train_b[is_joint]
    if u["tune"] == "per_replicate":
        model, _ = _fit_ml(spec, ml, feats, {**cfg, "n_jobs": 1}, seed)
        return model, feats
    est = build_estimator(spec, seed, cfg)
    if fixed_params:
        est.set_params(**fixed_params)
    return est.fit(ml[feats], ml[targets]), feats


def _replicate_fold(b, fold_id, label, train, test, specs, cfg, u, best_params):
    """Jedna replikacja jednego folda (fold_id = 0: dopasowanie do całego zbioru, bez testu)."""
    seed = cfg["seed"]
    targets = get_targets(cfg)
    rng = np.random.default_rng([seed, b, fold_id])
    train_b = _block_resample(train, u["block"], rng)
    rows, coefs = [], []
    for spec in specs:
        row = {"replicate": b, "fold": fold_id, "test_group": label, "model": spec.name}
        try:
            model, feats = _fit_one(spec, train_b, cfg, u, best_params.get((fold_id, spec.name)), seed)
        except NotIdentifiableError as e:
            rows.append({**row, "status": "not_identifiable", "note": str(e)})
            continue
        row["status"] = "ok"
        if not test.empty:
            data = test if spec.kind == "physical" else test[feats]
            row.update(evaluate(test[targets], np.asarray(model.predict(data)), targets))
        rows.append(row)
        coefs += [{"replicate": b, "fold": fold_id, "test_group": label, "model": spec.name, **c}
                  for c in model_coefficients(model, feats, targets)]
    return rows, coefs


def _fold_weights(n_folds, b, seed, resample):
    if not resample:
        return np.ones(n_folds)
    rng = np.random.default_rng([seed, b, 10 ** 6])
    return np.bincount(rng.integers(0, n_folds, size=n_folds), minlength=n_folds).astype(float)


def aggregate_replicates(raw, fold_ids, models, ref, targets, seed, resample_test_folds):
    """
    RMSE / MAE każdego modelu w każdej replikacji = średnia ważona po foldach (wagi z losowania
    foldów testowych); różnica względem odniesienia liczona na foldach wspólnych dla pary.
    """
    ok = raw[(raw["status"] == "ok") & raw["fold"].isin(fold_ids)]
    pos = {f: i for i, f in enumerate(fold_ids)}
    metrics = [f"{t}_{m}" for t in targets for m in ["rmse", "mae", "mse", "bias"]]
    rows = []
    for b, rep in ok.groupby("replicate"):
        w_all = _fold_weights(len(fold_ids), b, seed, resample_test_folds)
        table = rep.set_index(["model", "fold"])
        ref_vals = table.loc[ref] if ref in table.index.get_level_values(0) else None
        for m in models:
            if m not in table.index.get_level_values(0):
                continue
            mv = table.loc[m]
            w = np.array([w_all[pos[f]] for f in mv.index])
            if w.sum() == 0:
                continue
            row = {"replicate": b, "model": m, "n_folds": int((w > 0).sum())}
            for c in metrics:
                row[c] = float(np.average(mv[c], weights=w))
            if ref_vals is not None and m != ref:
                common = mv.index.intersection(ref_vals.index)
                wc = np.array([w_all[pos[f]] for f in common])
                for t in targets:
                    c = f"{t}_rmse"
                    row[f"{t}_rmse_diff_vs_ref"] = (float(np.average(mv.loc[common, c] - ref_vals.loc[common, c],
                                                                     weights=wc)) if wc.sum() > 0 else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_replicates(per_rep, models, ref, targets, ci, point=None):
    """Mediana i przedział percentylowy ci po replikacjach; P_better = odsetek replikacji z diff < 0."""
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    cols = [f"{t}_rmse" for t in targets] + [f"{t}_mae" for t in targets] + \
           [f"{t}_rmse_diff_vs_ref" for t in targets]
    rows = []
    for m in models:
        r = per_rep[per_rep["model"] == m]
        if r.empty:
            continue
        row = {"model": m, "reference": ref, "n_replicates": len(r)}
        for c in cols:
            if c not in r or r[c].isna().all():
                continue
            v = r[c].dropna()
            row[f"{c}_median"] = v.median()
            row[f"{c}_lo"] = v.quantile(lo)
            row[f"{c}_hi"] = v.quantile(hi)
            if c.endswith("diff_vs_ref"):
                row[f"{c}_P_better"] = float((v < 0).mean())
        if point is not None and m in set(point["model"]):
            p = point.set_index("model").loc[m]
            for t in targets:
                row[f"{t}_rmse_point"] = p.get(f"{t}_rmse_mean", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def compact_uncertainty(summary, targets, ci):
    """'mediana [lo; hi]' dla RMSE i różnic względem odniesienia - tabela do sprawozdania."""
    pct = int(round(ci * 100))
    out = summary[["model", "reference", "n_replicates"]].copy()
    for t in targets:
        for c, name in [(f"{t}_rmse", f"RMSE {t}"), (f"{t}_rmse_diff_vs_ref", f"dRMSE {t} vs ref")]:
            if f"{c}_median" not in summary:
                continue
            out[f"{name} (mediana [{pct}%])"] = [
                "" if pd.isna(m) else f"{m:.3f} [{l:.3f}; {h:.3f}]"
                for m, l, h in zip(summary[f"{c}_median"], summary[f"{c}_lo"], summary[f"{c}_hi"])]
            if f"{c}_P_better" in summary:
                out[f"P(lepszy) {t}"] = summary[f"{c}_P_better"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    return out


def plot_uncertainty(per_rep, models, ref, targets, ci, out_path):
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    models = [m for m in models if m in set(per_rep["model"])]
    fig, axes = plt.subplots(2, len(targets), figsize=(6.5 * len(targets), 0.45 * len(models) * 2 + 3),
                             squeeze=False)
    for j, t in enumerate(targets):
        for i, (col, title) in enumerate([(f"{t}_rmse", f"RMSE {t} [{TARGET_UNITS[t]}]"),
                                          (f"{t}_rmse_diff_vs_ref", f"RMSE {t} - RMSE {ref} [{TARGET_UNITS[t]}]")]):
            ax = axes[i, j]
            ms = [m for m in models if col in per_rep and per_rep.loc[per_rep["model"] == m, col].notna().any()]
            for k, m in enumerate(ms):
                v = per_rep.loc[per_rep["model"] == m, col].dropna()
                med, l, h = v.median(), v.quantile(lo), v.quantile(hi)
                ax.errorbar(med, k, xerr=[[med - l], [h - med]], fmt="o", color="C0" if m != ref else "C3", capsize=3)
            ax.set_yticks(range(len(ms)))
            ax.set_yticklabels(ms)
            ax.invert_yaxis()
            if i == 1:
                ax.axvline(0, color="k", lw=0.8)
            ax.set_title(f"{title}: mediana i {int(round(ci * 100))}% przedział")
            ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run_uncertainty(df, cfg, result, info, out_dir):
    u = uncertainty_config(cfg)
    B = int(u["n_replicates"])
    if B <= 0:
        return None
    targets = get_targets(cfg)
    models = u["models"] or cfg["models"]
    ref = reference_model(u, cfg, models)
    if ref not in models:
        models = [ref] + list(models)
    specs = [get_model_spec(m, cfg.get("grids", {}).get(m)) for m in models]
    if u["block"] not in df.columns:
        raise ValueError(f"uncertainty.block = {u['block']!r}: brak kolumny w danych")

    folds = result["fold_defs"]
    fold_ids = list(info["common_folds"])
    best_params = dict(result.get("best_params", {}))
    tasks = [(fid, g, df[~np.asarray(mask)], df[np.asarray(mask) & (df["is_joint_regression"] == True).to_numpy()])
             for fid, (g, mask) in enumerate(folds, start=1) if fid in fold_ids]
    if u["full_data_fit"]:
        if u["tune"] == "fixed":
            # hiperparametry dla dopasowania do całego zbioru: jedno strojenie na pełnych danych
            joint = df[df["is_joint_regression"] == True]
            for spec in specs:
                if spec.kind != "physical" and spec.grid:
                    _, bp = _fit_ml(spec, joint, spec.features or cfg["features"], cfg, cfg["seed"])
                    best_params[(0, spec.name)] = {k: v for k, v in bp.items() if not k.startswith("_")}
        tasks.append((0, "all", df, df.iloc[0:0]))

    print(f"\n>>> NIEPEWNOŚĆ (pkt 10): {B} replikacji x {len(tasks)} dopasowań, blok = {u['block']}, "
          f"tune = {u['tune']}, odniesienie = {ref}")
    cfg_w = copy.deepcopy(cfg)
    jobs = [(b, fid, g, tr, te) for b in range(B) for fid, g, tr, te in tasks]
    out = Parallel(n_jobs=u["n_jobs"])(
        delayed(_replicate_fold)(b, fid, g, tr, te, specs, cfg_w, u, best_params)
        for b, fid, g, tr, te in tqdm(jobs, desc="replikacje"))

    raw = pd.DataFrame([r for rows, _ in out for r in rows])
    coefs = pd.DataFrame([c for _, cs in out for c in cs])
    raw.to_csv(os.path.join(out_dir, "uncertainty_replicates_raw.csv"), index=False)
    if not coefs.empty:
        coefs.to_csv(os.path.join(out_dir, "uncertainty_coefficients_raw.csv"), index=False)

    per_rep = aggregate_replicates(raw, fold_ids, models, ref, targets, cfg["seed"], u["resample_test_folds"])
    per_rep.to_csv(os.path.join(out_dir, "uncertainty_replicates.csv"), index=False)
    summary = summarize_replicates(per_rep, models, ref, targets, u["ci"], info.get("summary_common"))
    pct = int(round(u["ci"] * 100))
    write_table(summary, out_dir, "uncertainty_summary",
                f"Replikacje (n = {B}, bootstrap blokowy {u['block']}): mediana i przedział {pct}% "
                f"dla RMSE, MAE i różnicy RMSE względem {ref}", 4)
    write_table(compact_uncertainty(summary, targets, u["ci"]), out_dir, "uncertainty_table",
                f"Niepewność wyników: mediana [{pct}% przedział] po {B} replikacjach; "
                f"dRMSE < 0 = model lepszy niż {ref}; P(lepszy) = odsetek replikacji z dRMSE < 0")
    status = raw.groupby(["model", "status"]).size().unstack(fill_value=0).reset_index()
    write_table(status, out_dir, "uncertainty_status", "Liczba dopasowań w replikacjach wg statusu")

    coef_sum = pd.DataFrame()
    if not coefs.empty:
        src = coefs[coefs["fold"] == 0] if u["full_data_fit"] else coefs
        coef_sum = coefficient_summary(src, ci=u["ci"])
        write_table(coef_sum, out_dir, "uncertainty_coefficients",
                    f"Współczynniki modeli: mediana i przedział {pct}% po replikacjach "
                    f"({'dopasowanie do bootstrapowej próby całego zbioru' if u['full_data_fit'] else 'wszystkie foldy'}); "
                    "CD-LIN: współczynniki przy m (wspólna) i d (różnicowa) w jednostkach cech", 6)
    plot_uncertainty(per_rep, models, ref, targets, u["ci"], os.path.join(out_dir, "uncertainty_rmse.png"))
    print(compact_uncertainty(summary, targets, u["ci"]).to_string(index=False))
    return {"summary": summary, "per_replicate": per_rep, "coefficients": coef_sum, "reference": ref,
            "ci": u["ci"]}
