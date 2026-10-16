# Etap 0 – naprawa i ujednolicenie pipeline'u

Dotyczy punktu 1 z listy prowadzącego na semestr 2 („sprawdzenie pipeline'u i rozwiązanie problemów”):
automatyczne generowanie plików i eksperymentów, rozbieżność RMSE dla tych samych modeli,
sposób liczenia R², liczba foldów i problem z foldem ΔT = 0.

## 1. Skąd brały się rozbieżności w raporcie z semestru 1

| Problem | Przyczyna | Naprawa |
|---|---|---|
| Różne RMSE tego samego modelu w różnych tabelach (np. SVR-RBF: 0.093 w tab. 3, 0.142 w `res_corr_temperature.csv`, 0.193 w tab. 8) | Tabele pochodziły z dwóch uruchomień (14.06 i 22.06) wykonanych z **różnymi siatkami hiperparametrów**, a liczby przepisywano ręcznie (literówki: GBR 2.944/2.994, XGBoost 1.739/1.139) | Jedno uruchomienie = jeden katalog z kompletem tabel (`.csv/.md/.tex`) generowanych automatycznie; konfiguracja i commit zapisywane razem z wynikami |
| Tabela 3 i tabela 8 identyczne, choć tylko tabela 8 miała być „bez ΔT = 0” | Obie pochodziły z `res_corr_temperature.csv` (8 foldów) | Każda tabela ma kolumnę `n_folds`; plik `fold_status` pokazuje, które foldy weszły do porównania |
| Wilcoxon p = 0.0078 dla prawie wszystkich par | To minimum osiągalne dla n = 8 foldów (2/2⁸) | W tabeli testów jest kolumna `min_attainable_p` |
| Bardzo dobry wynik AN-BL na foldzie ΔT = 0 (RMSE p = 0.069 MPa) | Gdy w treningu nie ma ΔT = 0, model używał **zakodowanych współczynników** dopasowanych wcześniej na całym zbiorze (przeciek) | Brak wartości domyślnych: fold dostaje status `not_identifiable`. Dodany wariant `AN-BL-NEAREST` z jawną regułą zastępczą (kalibracja ciśnieniowa z najniższej dostępnej temperatury) |
| R² „(0.000)” we wszystkich tabelach | Średnią R² zastępowano R² liczonym globalnie, a odchylenie ustawiano na 0 | R² per fold tylko tam, gdzie cel zmienia się w foldzie (dla ΔT w LOTO nie ma sensu – w foldzie jest jedna temperatura); R² pooled w osobnej, podpisanej tabeli `pooled_metrics` |
| Kolumna `params` w `folds.csv` / `res_*.csv` | `Counter(values)` liczony na zmiennej z poprzedniej metryki (`dT_maxae`), a nie na parametrach | Najlepsze parametry zapisywane per fold w `folds.csv` (`best_params`) |
| `predictions.csv` bez foldu 1 | Na sztywno `if fold == 0: continue` | Zapisywane są wszystkie foldy; o wykluczeniu z porównania decyduje status modelu |
| Kryterium strojenia S | SD liczone na foldzie walidacyjnym, nazwy kolumn zamienione (T/p) | SD z części treningowej folda zewnętrznego (zgodnie z pkt 12.4), jawna kolejność `[pressure, dT]` |
| `Pdir` | Liczony z różnicy **posortowanych etykiet** ciśnienia, nie z kolejności pomiarów w czasie – koduje etykietę (przeciek celu) | Cecha usunięta; ablacja „10 cech + Xinter/Pdir” zastąpiona wariantem z samym `Xinter` |
| Residual Drift (RD) | Modele w `repeatability_analysis.py` trenowano na wszystkich temperaturach, także tej, z której pochodził testowany `zero_end` | RD liczone z predykcji **out-of-fold**: `zero_end` z danej temperatury przewiduje model folda, w którym ta temperatura była testowa |
| Rozszerzone modele (GBR, HGBR, XGBoost, GPR, KRR, MLP) | Usunięte z repo dzień przed oddaniem raportu, więc tabel 3–6 nie da się odtworzyć | Decyzja zespołu: modele pozostają usunięte (niepotrzebne w semestrze 2) |
| `clean_roi` zwracała 2 albo 3 wartości | Wywołujący rozpakowywał 3–4 wartości; próbki wypadały po cichu przez `try/except` | Zawsze `(roi, quality, ratio)` |
| `requirements.txt` | Brak nowej linii: `seabornxgboost==3.2.0` | Poprawione, dodany `jinja2` (eksport LaTeX), usunięty `xgboost` |

Usunięte skrypty, których funkcje przejął pipeline: `baselines/run_cv.py`, `baselines/run_experiment.py`,
`baselines/utils/*`, `model_gain.py`, `repeatability_analysis.py`, `validation_diagnostics.py`,
`plot_model_results.py`, `stat_test.py`, `stat-2.py`, `build_global_db.py` (martwy kod).

## 2. Jak teraz uruchamiać

```
python main.py setup            # przygotowanie danych
python main.py run ablations    # eksperyment główny + ablacje
```

Każde uruchomienie tworzy `Output_files/runs/<data>_<nazwa>/` – szczegóły w README.

## 3. Wyniki po poprawkach

Uruchomienie `python main.py run ablations` (commit bazowy 72ebdd3 + zmiany etapu 0, seed 42).

### 3.1 Eksperyment główny (LOTO, 8 wspólnych foldów; fold dT = 0 wykluczony, bo AN-BL jest tam N/A)

| Model | p RMSE [MPa] | T RMSE [°C] | G_p [%] | G_T [%] |
|---|---|---|---|---|
| AN-BL | 0.142 (0.051) | 0.140 (0.091) | – | – |
| MO-LR | 0.075 (0.031) | 0.149 (0.078) | 47.1 | −6.3 |
| POLY2-RIDGE | 0.092 (0.031) | 0.136 (0.062) | 35.2 | 2.8 |
| SVR-RBF | 0.094 (0.045) | 0.244 (0.205) | 33.6 | −73.7 |
| NYSTROEM-SVR | 0.160 (0.123) | 0.250 (0.231) | −12.7 | −78.2 |
| RF | 2.415 (0.266) | 3.198 (0.924) | −1598.9 | −2177.6 |

AN-BL, MO-LR i POLY2-RIDGE odtwarzają raport co do 3. miejsca po przecinku. SVR-RBF = 0.094 zgadza się z tab. 3
raportu (uruchomienie 22.06), a nie z `res_corr_temperature.csv` (0.142, uruchomienie 14.06) – potwierdza to, że
rozbieżność wynikała ze zmiany siatek między uruchomieniami.

### 3.2 Fold dT = 0 (wcześniej ukryty przez „wyniki bez folda 1”)

| Model | p RMSE [MPa] | T RMSE [°C] |
|---|---|---|
| AN-BL | N/A (brak kalibracji ciśnieniowej) | N/A |
| AN-BL-NEAREST | 0.183 | 0.285 |
| MO-LR | 0.093 | 0.567 |
| POLY2-RIDGE | 0.358 | 0.392 |
| SVR-RBF | 0.766 | 4.384 |
| NYSTROEM-SVR | 0.546 | 3.903 |
| RF | 3.729 | 16.235 |

dT = 0 to brzeg zakresu temperatur, więc fold wymaga ekstrapolacji. Usunięcie go z tabel ukrywało nie tylko
problem AN-BL, ale też to, że modele nieliniowe (SVR, Nyström, RF) ekstrapolują wyraźnie gorzej niż liniowe.
Poprzednio raportowane 0.069 MPa dla AN-BL na tym foldzie wynikało z przecieku (współczynniki domyślne).

### 3.3 Ablacje (MO-LR, LOTO, 9 foldów)

| Wariant | p RMSE | T RMSE |
|---|---|---|
| A1 base4 (mu_X, mu_Y, diff, mean) | 0.082 | 0.202 |
| A1 ext8 (+ std, iqr) | 0.077 | 0.196 |
| A1 ext8 + Xinter | 0.102 | 0.207 |
| A2 tylko X | 3.077 | 1.621 |
| A2 tylko Y | 2.884 | 4.025 |
| A3 złe etykiety + zero_end w treningu | 0.742 | 0.191 |
| A4 dobre etykiety + zero_end w treningu | 0.078 | 0.191 |

`Xinter` pogarsza wynik modelu liniowego. Efekt samej korekty etykiet to porównanie A3 vs A4
(0.742 vs 0.078 MPa); A4 ≈ A1 ext8, czyli samo dodanie zero_end do treningu nic nie zmienia.

### 3.4 Powtarzalność

Dryft sygnału zgadza się z tab. 10 raportu. Nowe kolumny: |Δd| ≤ 0.06 GHz przy |Δm| do 0.85 GHz – dryft jest
praktycznie w całości w składowej wspólnej, więc wpływa na estymację temperatury, nie ciśnienia.
Residual Drift z predykcji out-of-fold: AN-BL 0.15, MO-LR 0.25, POLY2-RIDGE 0.24 (średnio); SVR-RBF i
NYSTROEM-SVR mają RD do ~5 przez `zero_end` z temperatur brzegowych.
