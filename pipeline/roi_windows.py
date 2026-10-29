"""
Cechy liczone w oknach ROI - eksperyment ze zmianą szerokości ROI.

Klucz konfiguracji "roi_window_cm":
    "full"  - jedno okno = całe ROI (8,5-12 m, 350 cm): dokładnie dzisiejszy paired_features.csv,
    L [cm]  - ROI dzielone na niepokrywające się okna o długości L (od ROI_MIN); z każdego okna
              liczone są te same cechy co z pełnego ROI (mu, std, IQR dla X i Y; m, d, Xinter).
              Każda para X/Y daje tyle wierszy, ile okien zmieściło się w ROI, więc modele dostają
              znacznie więcej danych do uczenia. Resztka ROI krótsza od L jest pomijana
              (np. L = 100 cm -> 3 okna = 300 cm).

Wszystkie okna jednej próbki mają te same etykiety i metadane (seria, warunek, powtórzenie), więc
schematy walidacji grupowej nigdy nie rozdzielą okien jednej próbki między trening i test.

Czyszczenie jest identyczne jak w build_folder_summary (utils.clean_roi: brzegi ROI, NaN, outliery
względem mediany całego ROI). Okno, w którym po czyszczeniu zostało mniej niż
roi_window_min_coverage typowej liczby punktów (w którymkolwiek kanale), jest pomijane.

Cache: Output_files/roi_windows/L{L}cm.npz - macierze (próbka x okno) średniej, std, IQR i liczby
punktów dla X i Y. Liczony raz, dla wszystkich długości z roi_window_cache_lengths naraz
(jeden odczyt plików). Nieaktualny cache (inna lista próbek lub ROI) jest przeliczany.
"""
import json
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from utils import ROI_MAX, ROI_MIN, clean_roi, read_measurement_file

FEATURES = ["mu_X", "mu_Y", "std_X", "std_Y", "iqr_X", "iqr_Y"]
ROI_CM = int(round((ROI_MAX - ROI_MIN) * 100))      # 350


def _window_stats(path, lengths_cm):
    """Dla jednego pliku: {L: (mean, std, iqr, count)} - tablice o długości liczby okien."""
    roi, _, _ = clean_roi(read_measurement_file(path))
    out = {}
    for L in lengths_cm:
        nw = ROI_CM // L
        mean, std, iqr = (np.full(nw, np.nan) for _ in range(3))
        cnt = np.zeros(nw)
        if roi is not None:
            z, v = roi["length"].to_numpy(), roi["value"].to_numpy()
            idx = np.floor((z - ROI_MIN) * 100 / L + 1e-9).astype(int)
            keep = (idx >= 0) & (idx < nw)
            idx, v = idx[keep], v[keep]
            bounds = np.searchsorted(idx, np.arange(nw + 1))      # z i idx są rosnące
            for w in range(nw):
                seg = v[bounds[w]:bounds[w + 1]]
                cnt[w] = len(seg)
                if len(seg):
                    mean[w], std[w] = seg.mean(), seg.std()
                    q75, q25 = np.percentile(seg, [75, 25])
                    iqr[w] = q75 - q25
        out[L] = (mean, std, iqr, cnt)
    return out


def _cache_path(cfg, L):
    return os.path.join(cfg["roi_window_cache_dir"], f"L{L}cm.npz")


def _cache_ok(cfg, L, sample_ids):
    path = _cache_path(cfg, L)
    if not os.path.exists(path):
        return False
    with np.load(path, allow_pickle=False) as c:
        meta = json.loads(str(c["meta"]))
        return list(c["sample_id"]) == list(sample_ids) and meta == _meta(L)


def _meta(L):
    return {"L_cm": int(L), "roi_min": ROI_MIN, "roi_max": ROI_MAX}


def build_cache(cfg, pf):
    """Liczy cache dla wszystkich długości z roi_window_cache_lengths (jeden odczyt plików)."""
    lengths = sorted(int(L) for L in cfg["roi_window_cache_lengths"])
    paths = pd.read_csv(cfg["training_dataset"]).set_index("sample_id")["path_x"]
    path_x = pf["sample_id"].map(paths)
    if path_x.isna().any():
        raise RuntimeError("Brak path_x dla części próbek - uruchom ponownie `python main.py setup`.")

    files = list(pf["sample_id"]) + list(path_x)
    gen = Parallel(n_jobs=cfg.get("n_jobs", -1), return_as="generator")(
        delayed(_window_stats)(f, lengths) for f in files)
    res = list(tqdm(gen, total=len(files), desc="Cechy w oknach ROI"))
    n = len(pf)
    os.makedirs(cfg["roi_window_cache_dir"], exist_ok=True)
    for L in lengths:
        arr = {name: np.array([r[L][i] for r in res]) for i, name in enumerate(["mean", "std", "iqr", "cnt"])}
        np.savez_compressed(
            _cache_path(cfg, L),
            sample_id=np.array(pf["sample_id"], dtype=str), meta=json.dumps(_meta(L)),
            **{f"{k}_Y": v[:n] for k, v in arr.items()}, **{f"{k}_X": v[n:] for k, v in arr.items()},
        )


def window_table(cfg, pf, L):
    """
    paired_features rozwinięty do poziomu okien: wiersz = (próbka, okno).
    Kolumny cech nadpisane wartościami z okna, dodane: window, z_center_m, window_cm.
    """
    L = int(L)
    if L <= 0 or L > ROI_CM:
        raise ValueError(f"roi_window_cm = {L}: dozwolone 1..{ROI_CM} albo 'full'")
    if L not in [int(x) for x in cfg["roi_window_cache_lengths"]]:
        cfg = dict(cfg, roi_window_cache_lengths=list(cfg["roi_window_cache_lengths"]) + [L])
    if not _cache_ok(cfg, L, pf["sample_id"]):
        print("Cache cech w oknach ROI nieaktualny lub brak - przeliczam (jeden odczyt wszystkich plików).")
        build_cache(cfg, pf)

    with np.load(_cache_path(cfg, L), allow_pickle=False) as c:
        data = {k: c[k] for k in c.files if k not in ("sample_id", "meta")}
    nw = data["mean_X"].shape[1]

    # pokrycie: liczba punktów w oknie względem typowej (mediana po pełnych oknach), osobno dla kanałów
    min_cov = cfg["roi_window_min_coverage"]
    valid = np.ones_like(data["cnt_X"], dtype=bool)
    for ch in ["X", "Y"]:
        cnt = data[f"cnt_{ch}"]
        typical = np.median(cnt[cnt > 0]) if np.any(cnt > 0) else 0
        valid &= cnt >= min_cov * typical
        valid &= np.isfinite(data[f"mean_{ch}"])

    i_s, i_w = np.nonzero(valid)
    out = pf.iloc[i_s].reset_index(drop=True)
    for ch in ["X", "Y"]:
        out[f"mu_{ch}"] = data[f"mean_{ch}"][i_s, i_w]
        out[f"std_{ch}"] = data[f"std_{ch}"][i_s, i_w]
        out[f"iqr_{ch}"] = data[f"iqr_{ch}"][i_s, i_w]
    out["m"] = (out["mu_X"] + out["mu_Y"]) / 2
    out["d"] = out["mu_X"] - out["mu_Y"]
    out["Xinter"] = out["mu_X"] * out["mu_Y"]
    out["window"] = i_w
    out["z_center_m"] = ROI_MIN + (i_w + 0.5) * L / 100
    out["window_cm"] = L

    dropped = 1 - valid.mean()
    print(f"Okna ROI: L = {L} cm, {nw} okien na próbkę, {len(out)} wierszy "
          f"({len(pf)} próbek; pominięto {dropped * 100:.2f}% okien z niepełnym pokryciem)")
    return out


def check_full_window(cfg, pf):
    """Kontrola: jedno okno 350 cm musi odtworzyć cechy z paired_features (te same punkty po czyszczeniu)."""
    t = window_table(cfg, pf, ROI_CM).set_index("sample_id")
    ref = pf.set_index("sample_id").loc[t.index]
    dev = {f: float(np.nanmax(np.abs(t[f].to_numpy() - ref[f].to_numpy()))) for f in FEATURES}
    return dev
