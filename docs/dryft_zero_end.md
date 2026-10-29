# Zadanie 9: powtarzalność i dryft zero_end

Uruchomienie:

```
python main.py run_drift        # configs/drift_zero_end.json, ok. 1,5 min
```

Wyniki: `Output_files/runs/20261029_174751_drift_zero_end/`, podkatalogi `condition`
(leave-one-condition-out) i `temperature` (leave-one-temperature-out, seria w teście jest całkowicie
niewidziana w treningu). Ta sama analiza powstaje automatycznie w **każdym** eksperymencie walidacyjnym
(pliki `repeatability_*`), kod: [pipeline/drift.py](../pipeline/drift.py).

## zero_end poza uczeniem

Pomiary `zero_end` nie biorą udziału w uczeniu żadnego modelu głównego:
- modele ML uczą się na rekordach `is_joint_regression`, które wykluczają `zero_end`;
- modele fizyczne kalibrują się na tych samych rekordach;
- przelicznik dryftu na T i p (AN-JOINT) też jest dopasowany bez `zero_end`;
- predykcje dla `zero_end` są wyłącznie out-of-fold.

Jedyny wyjątek to ablacja 4 (`include_zero_end_train`), która celowo pokazuje skutek takiego błędu.

## Jak wygląda seria pomiarowa

Z czasów akwizycji w nagłówkach plików OBR (`repeatability_series_sequence.md`) wynika, że seria to **jeden
przebieg ciśnienia trwający 36–55 min**: 10 szybkich pomiarów `zero_start` (co ~6 s), kolejne poziomy
ciśnienia (po 10 pomiarów), na końcu 10 pomiarów `zero_end`. Numer powtórzenia (`rep_idx`) to numer pomiaru
w bloku, a nie osobny cykl ciśnienia. **Dryft oznacza tu zmianę sygnału przy 0 MPa między początkiem a końcem
serii.**

`zero_end` jest tylko w 6 z 9 serii. Serie 16, 19 i 20 (ΔT = 27,4; 20,5; 24,7) mają przebieg malejący
10 → 0 MPa, a pomiar 0 MPa wykonano na końcu, więc nie ma pary start–koniec.

| seria | ΔT [°C] | przebieg | ciśnienie przed zero_end | czas serii [min] |
|---|---|---|---|---|
| 15 | 0 | rosnący 0 → 10 | 10 MPa | 36 |
| 2 | 16,2 | rosnący | 10 MPa | 42 |
| 21 | 33 | rosnący (przerwa 20 min przy 6–7 MPa) | 10 MPa | 55 |
| 25 | 37,5 | 0 → 10 → malejący | 0,1 MPa | 43 |
| 26 | 41,8 | 0 → 10 → malejący | 0,1 MPa | 44 |
| 27 | 45,7 | 0 → 10 → malejący | 0,1 MPa | 38 |

## Parametry każdej serii

Pełna tabela: `repeatability_series_parameters.md`. Dla każdej serii liczone są średnia i odchylenie z 10 par
`zero_end` − `zero_start`. Parametry z maila prowadzącego (kolumny `dX0`, `dY0`, `dm0`, `dd0`):

```
ΔX0 = X_zero_end − X_zero_start        Δm0 = (ΔX0 + ΔY0) / 2   (składowa wspólna)
ΔY0 = Y_zero_end − Y_zero_start        Δd0 = ΔX0 − ΔY0         (składowa różnicowa)
```

X i Y to średnie przesunięcie widmowe w ROI kanału X i Y [GHz]. Dodatkowo tabela zawiera:
- dryft przeliczony na temperaturę i ciśnienie (ΔT_drift, Δp_drift) przez odwrócenie modelu fizycznego,
- tempo dryftu na godzinę,
- szum krótkoterminowy (odchylenie w bloku `zero_start`).

| seria | ΔT | ΔX0 [GHz] | ΔY0 [GHz] | Δm0 [GHz] | Δd0 [GHz] | ΔT_drift [°C] | Δp_drift [MPa] |
|---|---|---|---|---|---|---|
| 15 | 0 | −0,277 | −0,283 | −0,280 | +0,007 | **+0,198** | +0,004 |
| 2 | 16,2 | −0,826 | −0,882 | −0,854 | +0,056 | **+0,589** | +0,024 |
| 21 | 33 | +0,389 | +0,408 | +0,398 | −0,019 | **−0,271** | −0,009 |
| 25 | 37,5 | −0,424 | −0,467 | −0,445 | +0,043 | **+0,298** | +0,018 |
| 26 | 41,8 | +0,002 | +0,000 | +0,001 | +0,001 | −0,001 | +0,001 |
| 27 | 45,7 | +0,106 | +0,124 | +0,115 | −0,018 | −0,075 | −0,007 |

## Odpowiedzi na pytania

### 1. Czy dryft jest głównie w składowej wspólnej, czy różnicowej?

**W składowej wspólnej.** Średnio |Δm0| = 0,283 GHz, a |Δd0| = 0,034 GHz, czyli 8 razy mniej. Na wykresie
Δm0–Δd0 (`repeatability_drift.png`) punkty leżą wzdłuż osi Δm0. Δd0 jest na poziomie szumu krótkoterminowego
(odchylenie d w bloku zero_start: 0,005–0,034 GHz).

Ponieważ składowa wspólna niesie temperaturę, dryft wygląda jak **zmiana temperatury włókna** w czasie serii,
a nie jak zmiana ciśnienia.

### 2. Czy dryft zależy od temperatury?

**Nie widać takiej zależności.** Spearman(ΔT, ΔT_drift) = −0,43 (p = 0,40), a dla |ΔT_drift| −0,54
(p = 0,27). Przy 6 seriach nie da się wykazać zależności. Dryft ma różne znaki niezależnie od temperatury
(+0,59 °C przy 16,2 °C, −0,27 °C przy 33 °C, ≈ 0 przy 41,8 °C).

### 3. Czy zależy od maksymalnego ciśnienia w serii?

**Na tych danych nie da się tego sprawdzić**, bo każda seria dochodzi do 10 MPa. Zamiast tego porównano
historię ciśnienia:
- przebieg rosnący (`zero_end` tuż po 10 MPa): średnio ΔT_drift = +0,17 °C, ale z dużym rozrzutem (+0,20; +0,59; −0,27);
- przebieg malejący (`zero_end` po 0,1 MPa): +0,07 °C (+0,30; 0,00; −0,08);
- czas trwania serii nie wpływa na wielkość dryftu: Spearman = 0,03.

Przebieg błędu w czasie (`repeatability_error_profile.md`) pokazuje coś ważniejszego. W seriach rosnących 2 i 21
błąd temperatury jest prawie stały przez cały przebieg, a **skacze dopiero przy `zero_end`**, po zrzuceniu
ciśnienia z 10 MPa: o +0,41 °C (seria 2) i −0,33 °C (seria 21). W seriach malejących skok przy końcu jest
bliski zera (≤ 0,02 °C). To wskazuje na **efekt odciążenia z wysokiego ciśnienia (histereza)**, a nie na dryft
narastający w czasie.

Dodatkowo w 5 z 6 serii błąd rośnie o +0,11…+0,19 °C między `zero_start` a pierwszym poziomem ciśnienia,
czyli przy pierwszym obciążeniu.

Rozstrzygnięcie wymagałoby serii z różnym ciśnieniem maksymalnym albo kilku cykli w jednej serii.

### 4. Czy dryft wpływa bardziej na temperaturę, czy na ciśnienie?

**Na temperaturę.** Względem błędu modelu odniesienia (AN-JOINT):

| | leave-one-condition-out | leave-one-temperature-out |
|---|---|---|
| \|ΔT_drift\| / RMSE_T | 1,99 | 1,34 |
| \|Δp_drift\| / RMSE_p | 0,22 | 0,16 |

Dryft temperatury jest porównywalny z błędem modelu, a nawet go przekracza. Dryft ciśnienia stanowi 15–20 %
błędu modelu.

Predykcje out-of-fold dla samych pomiarów `zero_end` (`repeatability_residual_drift_summary.md`) pokazują to
samo: średni |błąd T| wynosi 0,11–0,26 °C (zależnie od modelu i schematu), a |błąd p| 0,06–0,12 MPa, czyli tyle
co zwykły błąd ciśnienia. Najmniejszy Residual Drift w leave-one-condition-out ma CD-POLY (0,13), a modele
fizyczne mają około 0,16.

### 5. Czy na podstawie dryftu można zdefiniować wskaźnik jakości lub ostrzeżenie?

Zaproponowany wskaźnik:

```
Q = max( |ΔT_drift| / RMSE_T , |Δp_drift| / RMSE_p )      Q > 1: ostrzeżenie,  Q > 2: predykcja niewiarygodna
```

Flagi dla każdej serii są w `repeatability_quality_flags.md`. W leave-one-condition-out serie 2, 15, 21 i 25
dostają flagę „niewiarygodna”, a 26 i 27 „ok”.

**Weryfikacja daje wynik negatywny.** Wskaźnik sprawdzono na dwa sposoby:
- **Korekta dryftu.** Zakładając, że dryft narasta liniowo w czasie, od predykcji pomiarów serii odjęto
  przewidywany offset. RMSE_T **wzrosło** o 56 % (condition) i 34 % (temperature), a poprawa wystąpiła tylko
  w 26–29 % par (seria, model). Pojedynczy przypadek, w którym błąd faktycznie narasta w czasie, to seria 15.
- **Związek Q z błędem.** Spearman(Q_T, RMSE_T pomiarów serii) = −0,09 (p = 0,87). Q nie mówi, w której
  serii model myli się bardziej. Seria 26 ma Q ≈ 0, a jedno z wyższych RMSE_T, bo jej błąd zależy od ciśnienia
  w środku przebiegu, a start i koniec się pokrywają.

**Wniosek:** różnica `zero_end` − `zero_start` mierzy stabilność punktu zerowego czujnika, w tym skok po
odciążeniu z 10 MPa. **Nie nadaje się jednak ani do korekty predykcji, ani jako wiarygodne ostrzeżenie dla
pozostałych pomiarów serii.** Sensowne użycie:
- raportować Q i Residual Drift jako miarę stabilności zera (wymaganą przez załącznik),
- traktować `zero_end` jako test kontrolny: duży błąd predykcji dla `zero_end` oznacza, że stan czujnika po
  cyklu się zmienił,
- do ostrzegania o niewiarygodnych predykcjach potrzebne są inne narzędzia, np. przedziały predykcji
  z zadania 10.

## Pliki wynikowe

| plik | zawartość |
|---|---|
| `repeatability_series_sequence.*` | przebieg serii: kierunek, ciśnienie przed zero_end, czas |
| `repeatability_series_parameters.*` | parametry dryftu każdej serii |
| `repeatability_quality_flags.*` | wskaźnik Q i flaga |
| `repeatability_error_profile.*` | błąd T w czasie serii: skok przy starcie, przy końcu, nachylenie |
| `repeatability_drift_correction.*` | RMSE przed/po liniowej korekcie dryftu (seria × model) |
| `repeatability_answers.*` | odpowiedzi na 5 pytań (generowane automatycznie) |
| `repeatability_residual_drift*.*` | błędy predykcji out-of-fold dla zero_end, Residual Drift |
| `repeatability_drift.png` | Δm vs Δd, ΔT_drift i Δp_drift wg serii |

## Ograniczenia

- Tylko 6 serii ma pary `zero_start` / `zero_end`, a każda ma jedną parę bloków. 10 par w bloku to szybkie
  powtórzenia (~6 s) i nie są niezależne, więc wszystkie testy statystyczne liczone są po seriach (n = 6).
- Maksymalne ciśnienie jest we wszystkich seriach takie samo (10 MPa).
