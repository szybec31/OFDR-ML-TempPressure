"""
Punkt 10: grupowa (blokowa) predykcja konforemna z predykcji out-of-fold.

Wynik ocena = |predykcja - prawda| dla każdego rekordu testowego. Predykcje pochodzą z walidacji
krzyżowej (każdy rekord przewidziany przez model, który nie widział jego bloku), więc są to
wyniki "cross-conformal / CV+" bez osobnego zbioru kalibracyjnego.

Dane są zależne (10 szybkich powtórzeń jednego warunku, przesunięcia całych serii), dlatego
jednostką wymienialności jest BLOK, a nie pojedyncza obserwacja:
    block = "fold"       - grupa testowa folda (domyślnie; LOCO: warunek (T, p), LOTO: temperatura),
    block = <kolumna>    - np. "series_id" albo "condition_id".

Kwantyl (pooling CDF, Dunn, Wasserman, Ramdas 2022, "Distribution-free prediction sets for
two-layer hierarchical models"): każdy z n bloków ma łączną wagę 1/(n+1), rozłożoną równo na
jego obserwacje, a waga 1/(n+1) trafia do +inf (nowy blok). Połowa szerokości przedziału
    q = inf{ t : F_w(t) >= 1 - alpha },   przedział: predykcja ± q.
Gdy (1 - alpha)(n + 1) > n, q = inf - przy tak małej liczbie bloków nie da się zagwarantować
pokrycia (np. 9 serii: 95% niemożliwe, 90% = największy wynik).

Pokrycie empiryczne liczone uczciwie metodą leave-one-block-out: q dla bloku b wyznaczane z
pozostałych n - 1 bloków, pokrycie = odsetek obserwacji bloku b w przedziale; raportowana
średnia po blokach (każdy blok z tą samą wagą) i najgorszy blok. Bloki, dla których q z n - 1
pozostałych jest nieskończone, są pomijane (pokrycie trywialne); ich liczbę pokazuje
n_blocks_lobo_finite (0 -> pokrycie nieoznaczone, NaN).
"""
import numpy as np
import pandas as pd

DEFAULT_ALPHAS = (0.10, 0.05)


def block_quantile(scores, blocks, alpha):
    """Ważony kwantyl (1 - alpha) wyników z korektą skończonej próby; blok = jednostka."""
    scores = np.asarray(scores, dtype=float)
    blocks = np.asarray(blocks)
    uniq, inv, counts = np.unique(blocks, return_inverse=True, return_counts=True)
    n = len(uniq)
    if n == 0 or (1 - alpha) * (n + 1) > n:
        return np.inf
    w = 1.0 / counts[inv] / (n + 1)
    order = np.argsort(scores)
    cdf = np.cumsum(w[order])
    k = np.searchsorted(cdf, 1 - alpha - 1e-12)
    return float(scores[order][k]) if k < len(order) else np.inf


def _scores(test, targets):
    return {t: (test[f"{t}_pred"] - test[t]).abs().to_numpy(dtype=float) for t in targets}


def conformal_intervals(predictions, models, targets, fold_ids=None, alphas=DEFAULT_ALPHAS, block="fold"):
    """
    Tabela przedziałów dla każdego modelu, celu i poziomu: n_blocks, q (połowa szerokości),
    width = 2q, pokrycie leave-one-block-out (średnie po blokach i minimalne), wykonalność.
    Druga zwracana tabela: pokrycie w każdym bloku.
    """
    test = predictions[predictions["subset"] == "test"]
    if fold_ids is not None:
        test = test[test["fold"].isin(fold_ids)]
    key = "test_group" if block == "fold" else block
    if key not in test.columns:
        raise ValueError(f"conformal.block = {block!r}: brak kolumny {key!r} w predykcjach")
    rows, per_block = [], []
    for m in models:
        t = test[test["model"] == m]
        if t.empty:
            continue
        b = t[key].astype(str).to_numpy()
        uniq = np.unique(b)
        for target, s in _scores(t, targets).items():
            for a in alphas:
                q = block_quantile(s, b, a)
                cov = []
                for g in uniq:
                    inb = b == g
                    q_g = block_quantile(s[~inb], b[~inb], a)
                    c = float(np.mean(s[inb] <= q_g))
                    if np.isfinite(q_g):          # q = inf daje trywialne pokrycie 1 - pomijane
                        cov.append(c)
                    per_block.append({"model": m, "target": target, "level": 1 - a, "block": g,
                                      "q_loo": q_g, "coverage": c, "n_points": int(inb.sum())})
                rows.append({
                    "model": m, "target": target, "level": 1 - a, "block": block,
                    "n_blocks": len(uniq), "n_points": len(s),
                    "half_width": q, "width": 2 * q,
                    "coverage_lobo_mean": float(np.mean(cov)) if cov else np.nan,
                    "coverage_lobo_min": float(np.min(cov)) if cov else np.nan,
                    "n_blocks_lobo_finite": len(cov),
                    "feasible": bool(np.isfinite(q)),
                })
    return pd.DataFrame(rows), pd.DataFrame(per_block)


def conformal_wide(table, targets):
    """Jeden wiersz na model: szerokości i pokrycia dla każdego celu i poziomu (do tabeli końcowej)."""
    if table.empty:
        return pd.DataFrame()
    out = {}
    for _, r in table.iterrows():
        lvl = int(round(r["level"] * 100))
        d = out.setdefault(r["model"], {"model": r["model"], "conformal_blocks": r["n_blocks"]})
        d[f"{r['target']}_PI{lvl}_width"] = r["width"]
        d[f"{r['target']}_PI{lvl}_coverage"] = r["coverage_lobo_mean"]
    return pd.DataFrame(list(out.values()))
