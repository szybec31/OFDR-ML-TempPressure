"""
Wybór podzbioru stanów kalibracyjnych (dT, p) - punkt 6 listy prowadzącego.

Stan = jedna kombinacja (T, p); wybranie stanu oznacza użycie WSZYSTKICH jego powtórzeń.
Każda funkcja dostaje tabelę stanów (kolumny T, pressure; indeks 0..N-1) i docelową liczbę
stanów n, a zwraca posortowaną tablicę indeksów wybranych stanów. Liczba faktycznie wybranych
stanów może się różnić od n (np. regularna siatka nie trafi w każde n, a w siatce są luki).
Kolumna T to temperatura WZGLĘDEM referencji (T_celu - T_ref), tak jak w modelu fizycznym -
ma to znaczenie dla planu D-optymalnego [T^2, T, p], który nie ma wyrazu wolnego.

Metody:
    stratified_random - losowanie warstwowe: kolejne stany wybierane tak, aby wyrównywać liczbę
                        wybranych stanów w każdym poziomie dT i w każdym poziomie p (jak hiperkostka
                        łacińska na siatce), remisy rozstrzygane losowo,
    regular_grid      - iloczyn kT równomiernie rozłożonych poziomów dT i kp poziomów p (kT, kp >= 2),
    min_max_mid       - stany narożne (n = 4), narożne + środek (n = 5), siatka 3x3 min/środek/max (n >= 9),
    maximin           - zachłanne maksymalizowanie minimalnej odległości między stanami
                        w znormalizowanej przestrzeni (dT, p),
    d_optimal_phys    - plan D-optymalny dla modelu fizycznego [dT^2, dT, p] (AN-JOINT),
    d_optimal_quad    - plan D-optymalny dla pełnej powierzchni 2. stopnia [1, dT, p, dT^2, dT*p, p^2].
"""
import numpy as np

DETERMINISTIC = {"regular_grid", "min_max_mid", "maximin", "d_optimal_phys", "d_optimal_quad"}


def _unit(states):
    """Współrzędne znormalizowane do [0, 1]."""
    t, p = states["T"].to_numpy(float), states["pressure"].to_numpy(float)
    span = lambda v: (v - v.min()) / (np.ptp(v) or 1.0)
    return np.column_stack([span(t), span(p)])


def stratified_random(states, n, rng):
    t, p = states["T"].to_numpy(), states["pressure"].to_numpy()
    lt = {v: i for i, v in enumerate(np.unique(t))}
    lp = {v: i for i, v in enumerate(np.unique(p))}
    it = np.array([lt[v] for v in t])
    ip = np.array([lp[v] for v in p])
    cnt_t, cnt_p = np.zeros(len(lt)), np.zeros(len(lp))
    free = np.ones(len(states), bool)
    chosen = []
    for _ in range(min(n, len(states))):
        score = np.where(free, cnt_t[it] + cnt_p[ip], np.inf)
        best = np.flatnonzero(score == score.min())
        k = rng.choice(best)
        chosen.append(k)
        free[k] = False
        cnt_t[it[k]] += 1
        cnt_p[ip[k]] += 1
    return np.sort(np.array(chosen))


def regular_grid(states, n):
    lt, lp = np.unique(states["T"]), np.unique(states["pressure"])
    nt, np_ = len(lt), len(lp)
    if n >= len(states):
        return np.arange(len(states))
    best = None
    for kt in range(2, nt + 1):
        for kp in range(2, np_ + 1):
            key = (abs(kt * kp - n), abs(kt / nt - kp / np_))
            if best is None or key < best[0]:
                best = (key, kt, kp)
    _, kt, kp = best
    pick = lambda levels, k: levels[np.unique(np.round(np.linspace(0, len(levels) - 1, k)).astype(int))]
    mask = states["T"].isin(pick(lt, kt)) & states["pressure"].isin(pick(lp, kp))
    return np.flatnonzero(mask.to_numpy())


def _nearest_unique(u, targets):
    chosen = []
    for tgt in targets:
        dist = np.linalg.norm(u - tgt, axis=1)
        dist[chosen] = np.inf
        chosen.append(int(np.argmin(dist)))
    return np.sort(np.unique(chosen))


def min_max_mid(states, n):
    u = _unit(states)
    corners = [(0, 0), (0, 1), (1, 0), (1, 1)]
    if n <= 4:
        targets = corners
    elif n < 9:
        targets = corners + [(0.5, 0.5)]
    else:
        targets = [(a, b) for a in (0, 0.5, 1) for b in (0, 0.5, 1)]
    return _nearest_unique(u, np.array(targets, float))


def maximin(states, n):
    u = _unit(states)
    n = min(n, len(states))
    # start: stan najdalszy od środka ciężkości (narożnik), potem zachłannie najdalszy od wybranych
    chosen = [int(np.argmax(np.linalg.norm(u - u.mean(axis=0), axis=1)))]
    dmin = np.linalg.norm(u - u[chosen[0]], axis=1)
    while len(chosen) < n:
        dmin[chosen] = -np.inf
        k = int(np.argmax(dmin))
        chosen.append(k)
        dmin = np.minimum(dmin, np.linalg.norm(u - u[k], axis=1))
    return np.sort(np.array(chosen))


def _basis(states, kind):
    # skalowanie bez przesunięcia: zachowuje przestrzeń rozpiętą przez [T^2, T, p]
    t = states["T"].to_numpy(float) / (np.abs(states["T"]).max() or 1.0)
    p = states["pressure"].to_numpy(float) / (np.abs(states["pressure"]).max() or 1.0)
    if kind == "phys":
        return np.column_stack([t ** 2, t, p])
    return np.column_stack([np.ones_like(t), t, p, t ** 2, t * p, p ** 2])


def _logdet(M, ridge):
    return np.linalg.slogdet(M + ridge * np.eye(M.shape[-1]))[1]


def d_optimal(states, n, kind="phys", ridge=1e-8, max_iter=100):
    """Zachłanny wybór + wymiany Fedorowa maksymalizujące log det(F^T F)."""
    F = _basis(states, kind)
    N, k = F.shape
    n = min(n, N)
    if n >= N:
        return np.arange(N)
    outer = np.einsum("ni,nj->nij", F, F)

    chosen = []
    M = np.zeros((k, k))
    for _ in range(n):
        free = np.setdiff1d(np.arange(N), chosen)
        scores = _logdet(M + outer[free], 1e-6)
        j = free[int(np.argmax(scores))]
        chosen.append(j)
        M = M + outer[j]

    chosen = np.array(chosen)
    current = _logdet(M, ridge)
    for _ in range(max_iter):
        free = np.setdiff1d(np.arange(N), chosen)
        # wszystkie pary (usuwany i, dodawany j)
        cand = M[None, None] - outer[chosen][:, None] + outer[free][None, :]
        scores = _logdet(cand, ridge)
        i, j = np.unravel_index(np.argmax(scores), scores.shape)
        if scores[i, j] <= current + 1e-10:
            break
        M = cand[i, j]
        current = scores[i, j]
        chosen[i] = free[j]
    return np.sort(chosen)


def select_states(states, method, n, rng=None):
    if method == "stratified_random":
        return stratified_random(states, n, rng)
    if method == "regular_grid":
        return regular_grid(states, n)
    if method == "min_max_mid":
        return min_max_mid(states, n)
    if method == "maximin":
        return maximin(states, n)
    if method == "d_optimal_phys":
        return d_optimal(states, n, "phys")
    if method == "d_optimal_quad":
        return d_optimal(states, n, "quad")
    raise ValueError(f"Nieznana metoda wyboru stanów: {method}")
