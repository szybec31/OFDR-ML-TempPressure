"""
Schematy podziału na foldy zewnętrzne.

Fold = (etykieta, maska testu). Wszystko, co nie jest w teście, jest w treningu.
Poziomy temperatury pochodzą z kolumny celu temperaturowego (cfg["temperature_target"]: dT lub Tp),
więc etykiety foldów to np. "dT=24.7" albo "Tp=50".

Schematy typu leave-one-group-out (każdy rekord jest testowy dokładnie raz):
    temperature - jeden poziom temperatury testowy
    series      - jedna seria testowa (w tych danych: 1 seria = 1 temperatura)
    pressure    - jeden poziom ciśnienia testowy, we wszystkich temperaturach
    condition   - jedna kombinacja (T, p) - GŁÓWNY schemat walidacji (python main.py run);
                  test interpolacyjny: ta sama temperatura i to samo ciśnienie występują w treningu
                  w innych kombinacjach, a rekordy testu pochodzą z serii obecnej w treningu

Schemat "block" - usunięcie fragmentu przestrzeni (T, p):
    każdy element listy cfg["blocks"] to osobny fold; test = rekordy, których temperatura i p leżą
    w zadanym prostokącie (granice włącznie, brak granicy = bez ograniczenia), trening = reszta.
    Granice temperatury: T_min / T_max w jednostkach celu temperaturowego; dT_min / dT_max są
    dozwolone tylko przy celu dT. Bloki mogą się nakładać i nie muszą pokrywać całego zbioru.
"""
import numpy as np
import pandas as pd

from .metrics import temperature_target

SPLIT_SCHEMES = ["block", "condition", "pressure", "series", "temperature"]
_EPS = 1e-9


def condition_labels(df, t_col):
    return t_col + "=" + df[t_col].astype(str) + "_p=" + df["pressure"].astype(str)


def _groups(df, scheme, t_col):
    if scheme == "temperature":
        return t_col + "=" + df[t_col].astype(str)
    if scheme == "series":
        return "series=" + df["series_id"].astype(str)
    if scheme == "pressure":
        return "p=" + df["pressure"].astype(str)
    if scheme == "condition":
        return condition_labels(df, t_col)
    raise ValueError(scheme)


def _group_folds(df, scheme, t_col):
    groups = _groups(df, scheme, t_col).rename("group")
    # kolejność foldów w raportach
    order_cols = ["pressure", t_col] if scheme == "pressure" else [t_col, "pressure"]
    order = df.groupby(groups)[order_cols].min().sort_values(order_cols).index
    folds = [(g, (groups == g).to_numpy()) for g in order]
    if scheme == "series":
        # etykieta z temperaturami serii - czytelniejsze tabele
        folds = [(f"{g} ({t_col}={','.join(f'{v:g}' for v in sorted(df.loc[m, t_col].unique()))})", m)
                 for g, m in folds]
        one_to_one = (df.groupby("series_id")[t_col].nunique().max() == 1
                      and df.groupby(t_col)["series_id"].nunique().max() == 1)
        if one_to_one:
            print("UWAGA: każda seria ma dokładnie jedną temperaturę i każda temperatura jedną serię - "
                  "leave-one-series-out daje te same foldy co leave-one-temperature-out.")
    return folds


def _block_bounds(b, t_col):
    if t_col != "dT" and ("dT_min" in b or "dT_max" in b):
        raise ValueError(f"Blok {b['name']}: dT_min / dT_max przy celu {t_col} - użyj T_min / T_max "
                         f"(w jednostkach {t_col}).")
    lo_t = b.get("T_min", b.get("dT_min"))
    hi_t = b.get("T_max", b.get("dT_max"))
    return lo_t, hi_t, b.get("p_min"), b.get("p_max")


def _block_label(b, t_col):
    def rng(lo, hi, unit):
        lo = "" if lo is None else f"{lo:g}"
        hi = "" if hi is None else f"{hi:g}"
        return f"[{lo};{hi}]{unit}"
    lo_t, hi_t, lo_p, hi_p = _block_bounds(b, t_col)
    return f"{b['name']}: {t_col}{rng(lo_t, hi_t, '°C')} p{rng(lo_p, hi_p, 'MPa')}"


def block_mask(df, b, t_col="dT"):
    lo_t, hi_t, lo_p, hi_p = _block_bounds(b, t_col)
    lo_t = -np.inf if lo_t is None else lo_t
    hi_t = np.inf if hi_t is None else hi_t
    lo_p = -np.inf if lo_p is None else lo_p
    hi_p = np.inf if hi_p is None else hi_p
    return (
        (df[t_col] >= lo_t - _EPS) & (df[t_col] <= hi_t + _EPS)
        & (df["pressure"] >= lo_p - _EPS) & (df["pressure"] <= hi_p + _EPS)
    ).to_numpy()


def _block_folds(df, blocks, t_col):
    if not blocks:
        raise ValueError('Schemat "block" wymaga listy "blocks" w konfiguracji.')
    folds = []
    for b in blocks:
        mask = block_mask(df, b, t_col)
        if not mask.any():
            raise ValueError(f"Blok {b['name']} nie zawiera żadnych rekordów.")
        folds.append((_block_label(b, t_col), mask))
    return folds


def make_folds(df, cfg):
    scheme = cfg["split"]
    t_col = temperature_target(cfg)
    if scheme == "block":
        return _block_folds(df, cfg.get("blocks"), t_col)
    if scheme in SPLIT_SCHEMES:
        return _group_folds(df, scheme, t_col)
    raise ValueError(f"Nieznany schemat podziału: {scheme}. Dostępne: {SPLIT_SCHEMES}")


def select_folds(df, cfg):
    """make_folds bez foldów z cfg["exclude_test_groups"]; nieznana etykieta = błąd (nie cicha zmiana)."""
    folds = make_folds(df, cfg)
    exclude = cfg.get("exclude_test_groups") or []
    labels = [g for g, _ in folds]
    unknown = [e for e in exclude if e not in labels]
    if unknown:
        raise ValueError(f"exclude_test_groups: {unknown} nie pasuje do żadnego folda. "
                         f"Dostępne etykiety: {labels}")
    return [(g, m) for g, m in folds if g not in exclude]


def split_diagnostics(df, folds, t_col="dT"):
    """Dla każdego folda: rozmiar testu i ile serii / temperatur / ciśnień dzielą train i test."""
    rows = []
    for label, mask in folds:
        test, train = df[mask], df[~mask]
        s_tr, s_te = set(train["series_id"]), set(test["series_id"])
        rows.append({
            "test_group": label,
            "test_size": len(test),
            "train_size": len(train),
            "test_series": ",".join(map(str, sorted(s_te))),
            "shared_series_count": len(s_tr & s_te),
            f"test_{t_col}_levels": ",".join(f"{v:g}" for v in sorted(test[t_col].unique())),
            "test_p_levels": ",".join(f"{v:g}" for v in sorted(test["pressure"].unique())),
            f"shared_{t_col}_count": len(set(train[t_col]) & set(test[t_col])),
            "shared_pressure_count": len(set(train["pressure"]) & set(test["pressure"])),
        })
    return pd.DataFrame(rows)
