"""Schematy podziału na foldy zewnętrzne (grupy LeaveOneGroupOut)."""
import pandas as pd

SPLIT_SCHEMES = {
    # jeden poziom temperatury testowy
    "temperature": lambda df: "dT=" + df["dT"].astype(str),
    # jedna seria pomiarowa testowa (w tych danych: 1 seria = 1 temperatura)
    "series": lambda df: "series=" + df["series_id"].astype(str),
    # jeden poziom ciśnienia testowy (we wszystkich temperaturach)
    "pressure": lambda df: "p=" + df["pressure"].astype(str),
    # jedna kombinacja (temperatura, ciśnienie) - test interpolacyjny, NIE test generalizacji
    "condition": lambda df: "dT=" + df["dT"].astype(str) + "_p=" + df["pressure"].astype(str),
}


def make_groups(df, scheme):
    if scheme not in SPLIT_SCHEMES:
        raise ValueError(f"Nieznany schemat podziału: {scheme}. Dostępne: {sorted(SPLIT_SCHEMES)}")
    return SPLIT_SCHEMES[scheme](df).rename("group")


def split_diagnostics(df, groups):
    """Dla każdego folda: ile serii / temperatur / ciśnień jest wspólnych dla train i test."""
    rows = []
    for g in sorted(groups.unique(), key=str):
        test = df[groups == g]
        train = df[groups != g]
        s_tr, s_te = set(train["series_id"]), set(test["series_id"])
        rows.append({
            "test_group": g,
            "test_size": len(test),
            "test_series": ",".join(map(str, sorted(s_te))),
            "shared_series_count": len(s_tr & s_te),
            "shared_dT_count": len(set(train["dT"]) & set(test["dT"])),
            "shared_pressure_count": len(set(train["pressure"]) & set(test["pressure"])),
        })
    return pd.DataFrame(rows)
