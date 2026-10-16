"""
Schematy podziału na foldy zewnętrzne.

Fold = (etykieta, maska testu). Wszystko, co nie jest w teście, jest w treningu.

Schematy typu leave-one-group-out (każdy rekord jest testowy dokładnie raz):
    temperature - jeden poziom dT testowy
    series      - jedna seria testowa (w tych danych: 1 seria = 1 temperatura)
    pressure    - jeden poziom ciśnienia testowy, we wszystkich temperaturach
    condition   - jedna kombinacja (dT, p); test interpolacyjny, NIE test generalizacji

Schemat "block" - usunięcie fragmentu przestrzeni (T, p):
    każdy element listy cfg["blocks"] to osobny fold; test = rekordy, których dT i p leżą
    w zadanym prostokącie (granice włącznie, brak granicy = bez ograniczenia), trening = reszta.
    Bloki mogą się nakładać i nie muszą pokrywać całego zbioru.
"""
import numpy as np
import pandas as pd

GROUP_SCHEMES = {
    "temperature": lambda df: "dT=" + df["dT"].astype(str),
    "series": lambda df: "series=" + df["series_id"].astype(str),
    "pressure": lambda df: "p=" + df["pressure"].astype(str),
    "condition": lambda df: "dT=" + df["dT"].astype(str) + "_p=" + df["pressure"].astype(str),
}

# kolejność foldów w raportach
GROUP_ORDER = {
    "temperature": ["dT", "pressure"],
    "series": ["dT", "pressure"],
    "pressure": ["pressure", "dT"],
    "condition": ["dT", "pressure"],
}

SPLIT_SCHEMES = sorted(list(GROUP_SCHEMES) + ["block"])
_EPS = 1e-9


def _group_folds(df, scheme):
    groups = GROUP_SCHEMES[scheme](df).rename("group")
    order = df.groupby(groups)[GROUP_ORDER[scheme]].min().sort_values(GROUP_ORDER[scheme]).index
    return [(g, (groups == g).to_numpy()) for g in order]


def _block_label(b):
    def rng(lo, hi, unit):
        lo = "" if lo is None else f"{lo:g}"
        hi = "" if hi is None else f"{hi:g}"
        return f"[{lo};{hi}]{unit}"
    return f"{b['name']}: dT{rng(b.get('dT_min'), b.get('dT_max'), '°C')} p{rng(b.get('p_min'), b.get('p_max'), 'MPa')}"


def block_mask(df, b):
    lo_t = -np.inf if b.get("dT_min") is None else b["dT_min"]
    hi_t = np.inf if b.get("dT_max") is None else b["dT_max"]
    lo_p = -np.inf if b.get("p_min") is None else b["p_min"]
    hi_p = np.inf if b.get("p_max") is None else b["p_max"]
    return (
        (df["dT"] >= lo_t - _EPS) & (df["dT"] <= hi_t + _EPS)
        & (df["pressure"] >= lo_p - _EPS) & (df["pressure"] <= hi_p + _EPS)
    ).to_numpy()


def _block_folds(df, blocks):
    if not blocks:
        raise ValueError('Schemat "block" wymaga listy "blocks" w konfiguracji.')
    folds = []
    for b in blocks:
        mask = block_mask(df, b)
        if not mask.any():
            raise ValueError(f"Blok {b['name']} nie zawiera żadnych rekordów.")
        folds.append((_block_label(b), mask))
    return folds


def make_folds(df, cfg):
    scheme = cfg["split"]
    if scheme in GROUP_SCHEMES:
        return _group_folds(df, scheme)
    if scheme == "block":
        return _block_folds(df, cfg.get("blocks"))
    raise ValueError(f"Nieznany schemat podziału: {scheme}. Dostępne: {SPLIT_SCHEMES}")


def split_diagnostics(df, folds):
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
            "test_dT_levels": ",".join(f"{v:g}" for v in sorted(test["dT"].unique())),
            "test_p_levels": ",".join(f"{v:g}" for v in sorted(test["pressure"].unique())),
            "shared_dT_count": len(set(train["dT"]) & set(test["dT"])),
            "shared_pressure_count": len(set(train["pressure"]) & set(test["pressure"])),
        })
    return pd.DataFrame(rows)
