"""
Punkt 6: ile stanów kalibracyjnych (dT, p) naprawdę potrzeba.

Schemat (wszystko wewnątrz zewnętrznej walidacji, test nigdy nie bierze udziału w wyborze stanów):
    dla każdego folda zewnętrznego (domyślnie leave-one-temperature-out):
        stany treningowe = kombinacje (dT, p) w rekordach is_joint_regression poza testem,
        dla każdej metody wyboru i każdej liczby stanów n: wybór podzbioru stanów,
            kalibracja (i strojenie hiperparametrów) wyłącznie na rekordach wybranych stanów,
            ocena na pełnym, niezmienionym zbiorze testowym folda.
    Metody losowe powtarzane n_repeats razy (różne ziarna) - raportowany jest rozrzut.

Agregacja: RMSE uśredniane po foldach (dla każdego powtórzenia), potem średnia / std / p10 / p90
po powtórzeniach. Próg "pogorszenie <= x%" liczony względem pełnej kalibracji tego samego modelu;
minimalna liczba stanów = najmniejsze n, dla którego warunek spełniają n i wszystkie większe plany
tej metody (odporne na niemonotoniczność krzywej).
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
from .calibration_design import DETERMINISTIC, select_states
from .cv import _fit_ml
from .metrics import TARGET_UNITS, evaluate, get_targets
from .models import get_model_spec
from .report import write_table
from .splits import select_folds


def _label(t):
    return f"RMSE {'p' if t == 'pressure' else t} [{TARGET_UNITS[t]}]"


def _size_levels(n_states, fractions, extra):
    """Lista (etykieta, docelowa liczba stanów) - ułamki pełnego planu + liczby bezwzględne."""
    levels = {}
    for f in fractions:
        levels.setdefault(max(1, int(round(f * n_states))), f"{round(f * 100):d}%")
    for n in extra:
        if n < n_states:
            levels.setdefault(int(n), f"n={int(n)}")
    return sorted(((lab, n) for n, lab in levels.items()), key=lambda x: -x[1])


def _designs_for_fold(states, cfg, fold_seed):
    sizes = _size_levels(len(states), cfg["fractions"], cfg["n_states_extra"])
    out = [("full", "100%", len(states), 0, np.arange(len(states)))]
    for method in cfg["methods"]:
        if method == "min_max_mid":
            for n in (4, 5, 9):
                out.append((method, f"n={n}", n, 0, select_states(states, method, n)))
            continue
        for label, n in sizes:
            if n >= len(states):
                continue
            if method in DETERMINISTIC:
                out.append((method, label, n, 0, select_states(states, method, n)))
            else:
                for rep in range(cfg["n_repeats"]):
                    rng = np.random.default_rng([fold_seed, n, rep])
                    out.append((method, label, n, rep, select_states(states, method, n, rng)))
    return out


def _evaluate_design(train, test, specs, cfg, seed):
    """Kalibracja wszystkich modeli na rekordach jednego planu i ocena na teście folda."""
    targets = get_targets(cfg)
    n_states = train.groupby([targets[1], "pressure"]).ngroups
    rows = []
    for spec in specs:
        row = {"model": spec.name, "n_states": n_states, "n_train": len(train)}
        if n_states < spec.min_states:
            row["status"] = "too_few_states"
            rows.append(row)
            continue
        feats = spec.features or cfg["features"]
        try:
            model, best = _fit_ml(spec, train, feats, cfg, seed)
            pred = np.asarray(model.predict(test[feats]))
            if not np.all(np.isfinite(pred)):
                raise NotIdentifiableError("niefizyczne / nieskończone predykcje")
        except (NotIdentifiableError, ValueError, np.linalg.LinAlgError) as e:
            row.update({"status": "not_identifiable", "note": str(e)[:120]})
            rows.append(row)
            continue
        row.update({"status": "ok", "best_params": str(best) if best else "", **evaluate(test[targets], pred, targets)})
        rows.append(row)
    return rows


def _task(key, train, test, specs, cfg, seed):
    return key, _evaluate_design(train, test, specs, cfg, seed)


def _aggregate(raw, models, targets):
    ok = raw[raw["status"] == "ok"]
    n_folds = raw["fold"].nunique()
    # grupowanie po etykiecie rozmiaru: "25%" to w różnych foldach nieco inna liczba stanów
    keys = ["model", "method", "size_label", "rep"]
    per_rep = ok.groupby(keys).agg(
        n_folds=("fold", "nunique"), n_states=("n_states", "mean"), size_target=("size_target", "mean"),
        **{f"{t}_{m}": (f"{t}_{m}", "mean") for t in targets for m in ["rmse", "mae"]},
    ).reset_index()
    # powtórzenie liczy się tylko, gdy model działał we wszystkich foldach
    per_rep = per_rep[per_rep["n_folds"] == n_folds]

    def q(x, v):
        return np.quantile(x, v) if len(x) else np.nan

    stats = {}
    for t in targets:
        c = f"{t}_rmse"
        stats |= {c: (c, "mean"), f"{c}_std": (c, "std"),
                  f"{c}_p10": (c, lambda x: q(x, 0.1)), f"{c}_p90": (c, lambda x: q(x, 0.9))}
    curve = per_rep.groupby(keys[:-1]).agg(
        n_reps=("rep", "count"), n_states=("n_states", "mean"), **stats,
    ).reset_index()

    full = curve[curve["method"] == "full"].set_index("model")
    for t in targets:
        curve[f"{t}_rel"] = curve[f"{t}_rmse"] / curve["model"].map(full[f"{t}_rmse"])
    curve["model"] = pd.Categorical(curve["model"], models, ordered=True)
    curve = curve.sort_values(["model", "method", "n_states"], ascending=[True, True, False])
    return per_rep, curve.reset_index(drop=True)


def _min_states(curve, thresholds, targets):
    """Najmniejsza liczba stanów, od której (włącznie z większymi planami) RMSE <= (1+x) * pełne."""
    rows = []
    full = curve[curve["method"] == "full"]
    for (model, method), g in curve[curve["method"] != "full"].groupby(["model", "method"], observed=True):
        g = pd.concat([g, full[full["model"] == model]]).sort_values("n_states", ascending=False)
        row = {"model": model, "method": method}
        for x in thresholds:
            rel_p, rel_t = (f"{t}_rel" for t in targets)
            for crit, cols in [("p", [rel_p]), ("T", [rel_t]), ("both", [rel_p, rel_t])]:
                ok = (g[cols] <= 1 + x + 1e-12).all(axis=1).to_numpy()
                # ciągły odcinek spełniający warunek, licząc od pełnego planu w dół
                k = len(ok) if ok.all() else int(np.argmin(ok))
                row[f"n_{crit}_{int(round(x * 100))}pct"] = g["n_states"].iloc[k - 1] if k > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- wykresy

METHOD_STYLE = {
    "full": dict(color="k", marker="*"),
    "stratified_random": dict(color="tab:blue", marker="o"),
    "regular_grid": dict(color="tab:orange", marker="s"),
    "min_max_mid": dict(color="tab:green", marker="^"),
    "maximin": dict(color="tab:red", marker="D"),
    "d_optimal_phys": dict(color="tab:purple", marker="v"),
    "d_optimal_quad": dict(color="tab:brown", marker="P"),
}


def _plot_model(curve, model, out_path, targets):
    c = curve[curve["model"] == model]
    full = c[c["method"] == "full"]
    if full.empty:
        return
    t_col = targets[1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, t in zip(axes, targets):
        ref = full[f"{t}_rmse"].iloc[0]
        for x, ls in [(0.05, ":"), (0.10, "--"), (0.20, "-.")]:
            ax.axhline(ref * (1 + x), color="grey", lw=0.8, ls=ls, label=f"pełna +{int(x * 100)}%")
        for method, g in c[c["method"] != "full"].groupby("method"):
            g = pd.concat([g, full]).sort_values("n_states")
            st = METHOD_STYLE.get(method, {})
            ax.plot(g["n_states"], g[f"{t}_rmse"], label=method, lw=1.3, ms=5, **st)
            if g["n_reps"].max() > 1:
                ax.fill_between(g["n_states"], g[f"{t}_rmse_p10"].fillna(g[f"{t}_rmse"]),
                                g[f"{t}_rmse_p90"].fillna(g[f"{t}_rmse"]), color=st.get("color"), alpha=0.15)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"liczba stanów kalibracyjnych ({t_col}, p)")
        ax.set_ylabel(_label(t))
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"{model}: RMSE w funkcji liczby stanów (średnia po foldach; pasmo = p10-p90 powtórzeń)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_method(curve, method, models, out_path, targets):
    t_col = targets[1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, t in zip(axes, targets):
        for model in models:
            c = curve[(curve["model"] == model) & curve["method"].isin([method, "full"])].sort_values("n_states")
            if c.empty:
                continue
            line, = ax.plot(c["n_states"], c[f"{t}_rmse"], marker="o", ms=4, label=model)
            if c["n_reps"].max() > 1:
                ax.fill_between(c["n_states"], c[f"{t}_rmse_p10"].fillna(c[f"{t}_rmse"]),
                                c[f"{t}_rmse_p90"].fillna(c[f"{t}_rmse"]), color=line.get_color(), alpha=0.15)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"liczba stanów kalibracyjnych ({t_col}, p)")
        ax.set_ylabel(_label(t))
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Porównanie modeli - wybór stanów: {method}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_designs(designs, out_path, t_col, sizes=(5, 12, 28)):
    """Przykładowe plany (pierwszy fold, pierwsze powtórzenie) dla kilku liczb stanów."""
    fold = designs["fold"].min()
    d = designs[(designs["fold"] == fold) & (designs["rep"] == 0)]
    full = d[d["method"] == "full"].iloc[0]["states"]
    methods = [m for m in d["method"].unique() if m != "full"]
    fig, axes = plt.subplots(len(sizes), len(methods), figsize=(2.6 * len(methods), 2.4 * len(sizes)),
                             squeeze=False, sharex=True, sharey=True)
    for j, method in enumerate(methods):
        dm = d[d["method"] == method]
        for i, n in enumerate(sizes):
            ax = axes[i, j]
            ax.scatter([s[0] for s in full], [s[1] for s in full], s=6, c="lightgrey")
            if dm.empty:
                continue
            row = dm.iloc[np.argmin(np.abs(dm["n_states"] - n))]
            ax.scatter([s[0] for s in row["states"]], [s[1] for s in row["states"]], s=18, c="tab:red")
            ax.set_title(f"{method}\nn={row['n_states']}", fontsize=8)
            if i == len(sizes) - 1:
                ax.set_xlabel(f"{t_col} [°C]", fontsize=8)
            if j == 0:
                ax.set_ylabel("p [MPa]", fontsize=8)
    fig.suptitle(f"Przykładowe plany kalibracyjne (fold {fold})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- uruchomienie

def run_reduced_calibration(cfg, out_dir, df):
    seed = cfg["seed"]
    targets = get_targets(cfg)
    t_col = targets[1]
    t_ref = cfg["temperature_reference"]
    specs = [get_model_spec(m, cfg.get("grids", {}).get(m)) for m in cfg["models"]]
    models = [s.name for s in specs]
    inner_cfg = copy.deepcopy(cfg)
    inner_cfg["n_jobs"] = 1                 # równolegle liczone są plany, nie siatka

    joint = df["is_joint_regression"] == True
    folds = select_folds(df, cfg)

    tasks, design_rows = [], []
    for fold_id, (g, mask) in enumerate(folds, start=1):
        train_all = df[~mask & joint.to_numpy()]
        test = df[mask & joint.to_numpy()]
        states = train_all.groupby([t_col, "pressure"]).size().reset_index(name="n_rows")
        # plan liczony w temperaturze względem referencji (jak model fizyczny)
        states["T"] = states[t_col] - t_ref
        sid = {s: i for i, s in enumerate(zip(states[t_col], states["pressure"]))}
        state_id = np.array([sid[s] for s in zip(train_all[t_col], train_all["pressure"])])

        cache = {}
        for method, label, n_target, rep, idx in _designs_for_fold(states, cfg, seed + fold_id):
            k = tuple(idx)
            design_rows.append({
                "fold": fold_id, "test_group": g, "method": method, "size_label": label,
                "size_target": n_target, "rep": rep, "n_states": len(idx), "design_key": (fold_id, k),
                # (temperatura w jednostkach celu, ciśnienie)
                "states": list(zip(states[t_col].iloc[idx].tolist(), states["pressure"].iloc[idx].tolist())),
            })
            if k not in cache:
                cache[k] = True
                rows = train_all[np.isin(state_id, idx)]
                tasks.append(((fold_id, k), rows, test))

    designs = pd.DataFrame(design_rows)
    print(f"Foldy: {len(folds)}, planów: {len(designs)}, unikalnych kalibracji: {len(tasks)}, "
          f"modele: {', '.join(models)}")

    parallel = Parallel(n_jobs=cfg.get("n_jobs", -1), return_as="generator_unordered")
    gen = parallel(delayed(_task)(key, tr, te, specs, inner_cfg, seed) for key, tr, te in tasks)
    results = {}
    for key, rows in tqdm(gen, total=len(tasks), desc="Kalibracje"):
        results[key] = rows

    raw = []
    for _, d in designs.iterrows():
        for r in results[d["design_key"]]:
            raw.append({k: d[k] for k in ["fold", "test_group", "method", "size_label", "size_target", "rep"]} | r)
    raw = pd.DataFrame(raw)

    per_rep, curve = _aggregate(raw, models, targets)
    thresholds = cfg.get("thresholds", [0.05, 0.10, 0.20])
    min_n = _min_states(curve, thresholds, targets)

    designs_out = designs.drop(columns="design_key").assign(states=lambda x: x["states"].map(str))
    designs_out.to_csv(os.path.join(out_dir, "designs.csv"), index=False)
    raw.to_csv(os.path.join(out_dir, "results_raw.csv"), index=False)
    per_rep.to_csv(os.path.join(out_dir, "results_per_repeat.csv"), index=False)

    show = ["model", "method", "size_label", "n_states", "n_reps"] + [
        f"{t}_{s}" for t in targets for s in ["rmse", "rmse_std", "rel"]]
    write_table(curve, out_dir, "curve_full")
    write_table(curve[show], out_dir, "curve",
                f"RMSE vs liczba stanów (średnia po {raw['fold'].nunique()} foldach; std po powtórzeniach; "
                f"rel = względem pełnej kalibracji)")
    write_table(min_n, out_dir, "min_states",
                "Minimalna liczba stanów, przy której RMSE pogarsza się nie więcej niż o x% (p, T, oba)", digits=1)

    status = raw.groupby(["model", "status"]).size().unstack(fill_value=0).reset_index()
    write_table(status, out_dir, "status_counts", "Liczba kalibracji wg statusu")

    for model in models:
        _plot_model(curve, model, os.path.join(out_dir, f"rmse_vs_states_{model}.png"), targets)
    for method in cfg["methods"]:
        _plot_method(curve, method, models, os.path.join(out_dir, f"models_{method}.png"), targets)
    _plot_designs(designs, os.path.join(out_dir, "designs_examples.png"), t_col)

    print("\n" + curve[curve["method"].isin(["full", "stratified_random", "d_optimal_phys"])][
        ["model", "method", "size_label", "n_states"] + [f"{t}_rmse" for t in targets]].to_string(index=False))
    print("\n" + min_n.to_string(index=False))
    return {"curve": curve, "min_states": min_n}
