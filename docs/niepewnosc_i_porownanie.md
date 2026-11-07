# Punkty 10 i 11: niepewność wyników i końcowe porównanie modeli

Uruchomienie: `python main.py run_final` (config [configs/final_comparison.json](../configs/final_comparison.json)).
Config ma cztery warianty: `condition` (schemat główny), `series`, `temperature` i `pressure`. Każdy wariant
zapisuje zwykłe wyniki walidacji oraz wyniki punktu 10, a w katalogu głównym uruchomienia powstaje tabela
końcowa z punktu 11.

Wszystkie elementy działają też w każdym innym configu CV:
- predykcja konforemna jest liczona zawsze (klucz `"conformal"`),
- replikacje są włączane kluczem `"uncertainty": {"n_replicates": N}`,
- tabela końcowa powstaje po dodaniu klucza `"final_table"` do configu z wariantami.

## Punkt 10a: replikacje z ponownym dopasowaniem (`pipeline/uncertainty.py`)

Jedna replikacja to bootstrap blokowy:

1. W każdym foldzie zewnętrznym dane treningowe są losowane ze zwracaniem **blokami**. Domyślny blok to
   warunek (T, p), czyli wszystkie 10 szybkich powtórzeń jednego stanu razem (`"block": "condition_id"`).
   Pojedynczych obserwacji nie losujemy, bo powtórzenia są silnie zależne. Wszystkie modele są dopasowywane
   od nowa, a test folda się nie zmienia.
2. Przy `"resample_test_folds": true` foldy testowe też są losowane ze zwracaniem i dają wagi w średniej
   RMSE. Ten sam los dotyczy wszystkich modeli, więc różnice między modelami są parowane.
3. Przy `"full_data_fit": true` każdy model jest też dopasowywany do bootstrapowej próby całego zbioru.
   Z tych dopasowań pochodzą przedziały współczynników.

Hiperparametry modeli ML (`"tune"`):
- `"fixed"` (domyślnie): biorą się ze zwykłej walidacji danego folda, a w replikacjach jest tylko ponowne
  dopasowanie. To szybkie, ale pomija niepewność samego wyboru hiperparametrów.
- `"per_replicate"`: pełne strojenie w każdej replikacji. Dokładniejsze, ale wielokrotnie wolniejsze.

Dla każdej replikacji zapisywane są:

| plik | zawartość |
|---|---|
| `uncertainty_replicates_raw.csv` | błędy każdego modelu w każdym foldzie każdej replikacji |
| `uncertainty_replicates.csv` | RMSE / MAE zagregowane po foldach i różnica RMSE względem modelu analitycznego (`*_rmse_diff_vs_ref`) |
| `uncertainty_coefficients_raw.csv` | współczynniki modeli (fold 0 = dopasowanie do całego zbioru) |

Raporty z medianą i przedziałem 95% (percentylowym):

| plik | zawartość |
|---|---|
| `uncertainty_table.md` | RMSE p, RMSE T, ΔRMSE względem AN-JOINT; `P(lepszy)` = odsetek replikacji, w których model ma niższe RMSE |
| `uncertainty_summary.*` | to samo w pełnej postaci (z MAE i estymatą punktową) |
| `uncertainty_coefficients.*` | współczynniki: AN-* (A, B, C dla X i Y), CD-LIN (wyraz wolny, współczynnik przy m i przy d, w jednostkach cech), CD-POLY / POLY2-RIDGE / MO-LR |
| `uncertainty_rmse.png` | mediana i przedział RMSE oraz ΔRMSE dla każdego modelu |
| `uncertainty_status.*` | ile dopasowań w replikacjach było N/A (np. AN-BL bez punktów kalibracyjnych) |

Jak czytać wyniki:
- Model jest wiarygodnie lepszy od analitycznego, jeżeli cały przedział ΔRMSE leży poniżej zera.
  Przedział obejmujący zero oznacza różnicę w granicach niepewności.
- Bootstrap treningu zostawia średnio około 63% różnych warunków (reszta to powtórzone kopie), więc RMSE
  w replikacjach bywa nieco wyższe niż w zwykłej walidacji. Przedział opisuje rozrzut, a do porównań służy
  ΔRMSE, liczone na tych samych losowaniach dla obu modeli.
- `"block": "series_id"` losuje całe serie. Przy 9 seriach (1 seria = 1 temperatura) usuwa to całe poziomy
  temperatury i często czyni modele analityczne nieidentyfikowalnymi, dlatego nie jest to wartość domyślna.

## Punkt 10b: grupowa predykcja konforemna (`pipeline/conformal.py`)

Liczona z predykcji out-of-fold każdego uruchomienia CV, bez dodatkowego uczenia. Wynikiem dla rekordu jest
|predykcja − prawda|. Jednostką wymienialności jest blok, domyślnie grupa testowa folda (`"block": "fold"`):
- w LOCO blokiem jest warunek (T, p), czyli 119 bloków,
- w LOTO / LOSO blokiem jest temperatura (seria), czyli 9 bloków,
- w LOPO blokiem jest poziom ciśnienia, czyli 14 bloków.

Kwantyl to pooling CDF z pracy Dunn, Wasserman, Ramdas (2022). Każdy blok ma tę samą łączną wagę, a jeden
blok „zapasowy” ma wagę +∞. Przedział ma postać predykcja ± q.

**Ograniczenie liczby bloków.** Przedział na poziomie 1 − α jest skończony tylko wtedy, gdy (1 − α)(n + 1) ≤ n:
- 90% wymaga co najmniej 9 bloków,
- 95% wymaga co najmniej 19 bloków.

Przy 9 seriach przedział 95% na poziomie serii jest więc nieskończony, a 90% jest równy największemu
wynikowi. To odpowiedź na warunek z instrukcji („jeżeli liczba niezależnych serii będzie wystarczająca”):
na poziomie serii **nie jest wystarczająca**. Wiarygodne przedziały daje schemat LOCO (119 bloków), z jednym
zastrzeżeniem: warunki z tej samej serii dzielą przesunięcie serii, więc dla zupełnie nowej sesji pomiarowej
pokrycie może być niższe od deklarowanego. Kontrolą jest pokrycie z LOTO.

Pokrycie empiryczne liczone jest metodą leave-one-block-out (`coverage_lobo_mean`, `coverage_lobo_min`).
Bloki, dla których q z pozostałych bloków jest nieskończone, są pomijane, bo ich pokrycie byłoby trywialnie
równe 1. Ich liczbę pokazuje `n_blocks_lobo_finite`.

Pliki: `conformal_intervals.*` (q, szerokość 2q, pokrycie, wykonalność) i `conformal_per_block.csv`.

## Punkt 11: tabela końcowa (`pipeline/final_table.py`)

Pliki `final_comparison.*` (pełna, CSV / MD / TeX) i `final_comparison_compact.md` (do sprawozdania)
w katalogu głównym uruchomienia. Skąd pochodzą kolumny:

| kolumna | źródło |
|---|---|
| RMSE i MAE T, MSE i MAE p, bias | schemat główny (`final_table.main`, domyślnie condition), średnia po foldach referencyjnych |
| max \|błąd\| | maksimum po wszystkich predykcjach testowych schematu głównego |
| przedział RMSE i ΔRMSE vs analityczny | replikacje schematu głównego (punkt 10a) |
| LOSO / LOTO / LOPO | RMSE p i T (średnia po foldach) i liczba foldów, w których model działał; przedziały z replikacji danego wariantu w pełnej tabeli |
| liczba parametrów | `count_parameters`: parametry dopasowywane na danych, bez standaryzacji cech; „nieparam.” = liczba rośnie z danymi (SVR: współczynniki dualne, GP: wagi, RF: węzły) |
| min. liczba stanów | z rejestru modeli (identyfikowalność); opcjonalnie empirycznie z `run_reduced` |
| PI90 / PI95 | ± połowa szerokości przedziału konforemnego z wariantu `final_table.conformal` |
| zero_end | średni \|błąd\| T i p na zero_end (predykcje out-of-fold) i Residual Drift (RD) w pełnej tabeli |

Empiryczna minimalna liczba stanów: w `final_table.reduced_calibration` podaj ścieżkę do `min_states.csv`
z uruchomienia `run_reduced`. Brana jest najmniejsza liczba stanów po wszystkich metodach wyboru, przy której
RMSE p i T pogarsza się o nie więcej niż `reduced_threshold` %.

W tych danych LOSO i LOTO mają identyczne foldy (1 seria = 1 temperatura), więc obie kolumny są takie same.
W configu replikacje dla wariantu `series` są przez to wyłączone.

## Czas obliczeń

Najdroższy jest wariant `condition`: 50 replikacji × 119 foldów × 11 modeli plus dopasowanie do całego
zbioru. Najwolniejsze są HYB-JOINT-GP (proces gaussowski z optymalizacją jądra) i SVR. Gdy czas jest
problemem, można:
- zmniejszyć `n_replicates`,
- ograniczyć listę modeli w replikacjach: `"uncertainty": {"models": [...]}`,
- wyłączyć replikacje w wariancie `condition` (`"n_replicates": 0`); przedziały konforemne zostaną, bo nie
  wymagają dodatkowego uczenia.
