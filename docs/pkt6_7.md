# Punkty 6 i 7: ograniczenie liczby stanów kalibracyjnych i przestrzenny charakter OFDR

Uruchomienie:

```
python main.py run_reduced    # punkt 6, configs/reduced_calibration.json (~25 min)
python main.py run_spatial    # punkt 7, configs/spatial_windows.json (~15 s; pierwszy raz ~1 min - budowa cache)
```

Oba eksperymenty używają schematu zewnętrznego leave-one-temperature-out (jak `run_temp`)
**bez folda dT = 0** (8 foldów, bo fold dT = 0 to ekstrapolacja poza zakres treningowy). Uwaga: główny
eksperyment (`run`) to teraz leave-one-condition-out; tu zostawiono temperatury, bo przy wyborze stanów
kalibracyjnych test na niewidzianej temperaturze jest trudniejszy i bardziej miarodajny. Można to zmienić kluczem
`exclude_test_groups` w configu.

## Nowe modele (pkt 4 listy – częściowo)

| Model | Opis | Cechy |
|---|---|---|
| AN-JOINT | ten sam model fizyczny co AN-BL, `mu_k = A_k dT² + B_k dT + C_k p`, ale wszystkie współczynniki dopasowane jednocześnie MNK na dowolnym zbiorze stanów (nie wymaga serii p = 0 i dT = 0) | mu_X, mu_Y |
| CD-LIN | regresja liniowa (p, dT) na składowej wspólnej i różnicowej | m, d |
| CD-POLY | wielomian 2. stopnia (ridge) na m, d | m, d |

Są też dodane do eksperymentów CV (`run`, `run_temp`, `run_series`, `run_pressure`, `run_block`).

---

## Punkt 6 – ograniczenie liczby stanów kalibracyjnych

### Metoda

- **Stan** = jedna kombinacja (dT, p); wybranie stanu = użycie wszystkich jego powtórzeń (~10).
  W foldzie treningowym jest 105–112 stanów.
- Stany wybierane są **tylko spośród stanów treningowych folda**; model (wraz ze strojeniem
  hiperparametrów, GroupKFold po seriach) kalibrowany jest wyłącznie na rekordach wybranych stanów;
  test jest zawsze pełny i niezmieniony.
- Rozmiary planu: 100, 75, 50, 33, 25 % oraz n = 20, 16, 12, 9, 6, 4, 3 stany. „Minimalny zestaw”
  zależy od modelu (`min_states` w `pipeline/models.py`): 3 dla AN-JOINT, CD-LIN, MO-LR; 6 dla modeli
  z członami 2. stopnia i SVR. Mniejsze plany dostają status `too_few_states`, plany, na których
  współczynników nie da się wyznaczyć (np. dwie temperatury, z których jedna to dT = 0 – kolumny dT²
  i dT są wtedy współliniowe) – `not_identifiable`.
- Metody wyboru (`pipeline/calibration_design.py`):
  - `stratified_random` – losowanie warstwowe: każdy kolejny stan wybierany tak, by wyrównywać liczbę
    stanów w każdym poziomie dT i każdym poziomie p (hiperkostka łacińska na siatce); **20 powtórzeń**,
  - `regular_grid` – iloczyn równomiernie rozłożonych poziomów dT i p,
  - `min_max_mid` – narożniki (4), narożniki + środek (5), siatka 3×3 min/środek/max (9),
  - `maximin` – zachłanne maksymalizowanie minimalnej odległości w znormalizowanej przestrzeni (dT, p),
  - `d_optimal_phys` – plan D-optymalny (maks. det FᵀF, zachłannie + wymiany Fedorowa) dla modelu
    fizycznego [dT², dT, p],
  - `d_optimal_quad` – jw. dla pełnej powierzchni 2. stopnia [1, dT, p, dT², dT·p, p²].
- Agregacja: RMSE średnio po 8 foldach dla każdego powtórzenia, potem średnia, std, p10, p90
  po powtórzeniach. **Minimalna liczba stanów dla progu x %** = najmniejsze n, przy którym
  RMSE ≤ (1 + x)·RMSE pełnej kalibracji tego samego modelu **dla tego n i wszystkich większych planów
  tej metody** (krzywe nie są monotoniczne, pojedynczy „szczęśliwy” plan się nie liczy).

### Wyniki

Uruchomienie `Output_files/runs/20261016_165021_reduced_calibration/` (8 foldów, 2101 unikalnych
kalibracji, ~25 min). Siatki hiperparametrów takie same jak w eksperymencie głównym – pełna kalibracja
odtwarza wyniki tabeli głównej (np. SVR-RBF 0.101 MPa / 0.240 °C).

Najważniejszy wykres: `models_stratified_random.png` (wszystkie modele, losowanie warstwowe, pasmo p10–p90
z 20 powtórzeń) oraz `rmse_vs_states_<model>.png` (jeden model, wszystkie metody wyboru).

**RMSE_p [MPa] / RMSE_T [°C], losowanie warstwowe** (średnia z 20 powtórzeń):

| stany | AN-JOINT | CD-LIN | MO-LR | CD-POLY | POLY2-RIDGE | SVR-RBF |
|---|---|---|---|---|---|---|
| 100 % (~106) | 0.078 / 0.148 | 0.077 / 0.148 | 0.075 / 0.149 | 0.079 / 0.150 | 0.089 / 0.140 | 0.101 / 0.240 |
| 50 % (~53) | 0.080 / 0.151 | 0.078 / 0.154 | 0.079 / 0.158 | 0.083 / 0.157 | 0.103 / 0.159 | 0.116 / 0.280 |
| 25 % (~26) | 0.082 / 0.156 | 0.079 / 0.156 | 0.084 / 0.168 | 0.088 / 0.165 | 0.157 / 0.239 | 0.141 / 0.325 |
| 12 | 0.087 / 0.157 | 0.083 / 0.167 | 0.103 / 0.206 | 0.102 / 0.197 | 0.225 / 0.333 | 0.247 / 0.442 |
| 9 | 0.092 / 0.168 | 0.086 / 0.171 | 0.133 / 0.247 | 0.151 / 0.304 | 0.268 / 0.433 | 0.302 / 0.652 |
| 6 | 0.103 / 0.173 | 0.099 / 0.217 | 0.220 / 0.357 | 0.93 / 2.25 | 1.94 / 2.76 | 0.50 / 1.33 |
| 4 | 0.187 / 0.307 | 0.143 / 0.250 | 0.242 / 0.387 | – | – | – |
| 3 | 0.444 / 0.658 | 0.331 / 0.573 | 0.618 / 0.487 | – | – | – |

Rozrzut między powtórzeniami (p10–p90) jest w `curve_full.csv`; np. AN-JOINT przy 9 stanach:
p 0.080–0.108 MPa, T 0.158–0.183 °C.

**Plan D-optymalny dla modelu fizycznego (`d_optimal_phys`)**:

| stany | AN-JOINT | CD-LIN | MO-LR | POLY2-RIDGE | SVR-RBF |
|---|---|---|---|---|---|
| 100 % | 0.078 / 0.148 | 0.077 / 0.148 | 0.075 / 0.149 | 0.089 / 0.140 | 0.101 / 0.240 |
| 9 | 0.077 / 0.151 | 0.087 / 0.147 | 0.095 / 0.181 | 0.258 / 0.370 | 0.218 / 0.427 |
| 6 | 0.081 / 0.150 | 0.085 / 0.168 | 0.096 / 0.221 | 0.759 / 0.927 | 0.227 / 0.941 |
| 4 | 0.083 / 0.153 | 0.082 / 0.134 | 0.092 / 0.160 | – | – |
| 3 | 0.091 / 0.191 | 0.096 / 0.148 | 0.096 / 0.204 | – | – |

**Minimalna liczba stanów** (pogorszenie obu RMSE ≤ x % względem pełnej kalibracji, dla tego planu i
wszystkich większych; pełna tabela: `min_states.md`):

| model | najlepsza metoda | ≤ 5 % | ≤ 10 % | ≤ 20 % | losowanie warstwowe ≤ 10 % / ≤ 20 % |
|---|---|---|---|---|---|
| AN-JOINT | d_optimal_phys | 52 | **4** | **4** | 16 / 9 |
| CD-LIN | d_optimal_phys | 35 | 12 | **4** | 16 / 9 |
| MO-LR | d_optimal_phys | 26 | 12 | 12 | 35 / 20 |
| CD-POLY | stratified_random | 106 | 35 | 16 | 35 / 16 |
| POLY2-RIDGE | stratified_random | 79 | 79 | 52 | 79 / 52 |
| SVR-RBF | d_optimal_quad | 79 | 35 | 35 | 106 / 52 |

### Wnioski z punktu 6

1. **Model fizycznie uwarunkowany potrzebuje rzędu kilku stanów.** AN-JOINT na planie D-optymalnym
   z **4 stanów** (3 temperatury, 2 ciśnienia: 0 i 10 MPa) ma 0.083 MPa / 0.153 °C, czyli +7 % / +3 %
   względem kalibracji na ~106 stanach. To redukcja liczby stanów ~25×. CD-LIN zachowuje się podobnie.
2. **Modele ML potrzebują dużo więcej danych.** POLY2-RIDGE i SVR-RBF mieszczą się w 20 % dopiero
   przy 35–52 stanach, a poniżej ~10 stanów ich błąd rośnie o rząd wielkości. Przy pełnej kalibracji
   ich przewaga i tak jest niewielka albo żadna – czyli dla tego czujnika struktura fizyczna jest
   „wartą danych” informacją a priori.
3. **Sposób wyboru ma znaczenie przy małej liczbie stanów.** Plan D-optymalny wybiera skrajne
   ciśnienia (0 i 10 MPa) i rozłożone temperatury – dokładnie to, czego potrzebuje model liniowy
   w p i kwadratowy w T. Maximin i regularna siatka wypadają gorzej: rozkładają stany równomiernie
   w przestrzeni, zamiast tam, gdzie są najbardziej informacyjne. Siatka 2×2 i min/max/mid przy
   dwóch temperaturach (w tym dT = 0) w ogóle nie wyznaczają członu dT² w AN-JOINT (`not_identifiable`).
4. **Próg 5 % jest na granicy szumu.** 5 % pełnego RMSE to ~0.004 MPa / ~0.007 °C, czyli mniej niż
   rozrzut między powtórzeniami losowania i między foldami; wyniki dla 5 % traktować ostrożnie,
   wnioski opierać na progach 10 % i 20 %.
5. **Uwaga praktyczna o koszcie kalibracji.** W praktyce kosztowna jest liczba **temperatur**
   (stabilizacja termiczna), a ciśnienie zmienia się szybko. Plan D-optymalny z 4 stanów używa
   3 temperatur, losowanie warstwowe z 9 stanów – 8 temperatur (bo wyrównuje liczność w każdym
   poziomie). Średnia liczba poziomów dT i p w planach jest w `designs.csv`. Jeżeli to ważne dla
   publikacji, można dodać wariant „mało temperatur × pełne ciśnienie” – osobny plan do ustalenia.

### Ograniczenia

- Plany wybierane są na siatce stanów, które fizycznie zmierzono; nie wiemy, jak wypadłby plan
  z punktami spoza siatki.
- Test to wciąż tylko 8 temperatur (8 foldów), więc różnice rzędu kilku procent nie są istotne
  statystycznie.
- Metody deterministyczne (siatka, maximin, D-optimal) mają jedno powtórzenie – brak rozrzutu.

---

## Punkt 7 – okna przestrzenne wzdłuż ROI

### Metoda

- Każdy plik X i Y czyszczony tak samo jak w `build_folder_summary` (`clean_roi`), potem punkty
  sumowane w binach 1 cm **po położeniu** (pliki mają różne siatki: 6137 vs 5624 punktów).
  Okno L = suma k sąsiednich binów, więc średnia w oknie jest dokładna. Kontrola: okno „full”
  odtwarza mu_X, mu_Y z `paired_features.csv` z dokładnością 3·10⁻¹⁴ GHz.
- Okna niepokrywające się, od 8.5 m: L = 1, 2, 5, 10, 25, 50, 100 cm i pełne ROI (350 cm).
  Dla L = 25 i 100 cm ostatnie 0.5 m ROI nie mieści się w całym oknie i jest pomijane.
- Okno z mniej niż 80 % typowej liczby punktów po czyszczeniu = brak sygnału (NaN).
- W każdym oknie cechy X, Y, m, d; modele AN-JOINT i CD-LIN w dwóch trybach:
  - **local** – osobne współczynniki dla każdego okna,
  - **global** – jeden zestaw współczynników dla całego włókna (MNK na złączonych oknach).
- RMSE(L) = RMSE po wszystkich (okno × rekord testowy) folda, średnio po foldach; dodatkowo
  mediana po oknach (odporna na pojedyncze złe okna).

### RMSE w funkcji długości okna

Średnia po 8 foldach (p [MPa] / T [°C]):

| L | okien | AN-JOINT local | AN-JOINT global | CD-LIN local | CD-LIN global |
|---|---|---|---|---|---|
| 1 cm | 350 | 0.147 / 0.191 | 0.455 / 0.365 | 0.173 / 0.291* | 0.453 / 0.370 |
| 2 cm | 175 | 0.139 / 0.186 | 0.446 / 0.358 | 0.159 / 0.288* | 0.445 / 0.362 |
| 5 cm | 70 | 0.124 / 0.177 | 0.397 / 0.327 | 0.117 / 0.173 | 0.396 / 0.331 |
| 10 cm | 35 | 0.102 / 0.167 | 0.261 / 0.245 | 0.210 / 0.446* | 0.262 / 0.249 |
| 25 cm | 14 | 0.090 / 0.162 | 0.154 / 0.200 | 0.089 / 0.160 | 0.154 / 0.202 |
| 50 cm | 7 | 0.085 / 0.159 | 0.117 / 0.179 | 0.085 / 0.158 | 0.117 / 0.180 |
| 100 cm | 3 | 0.081 / 0.157 | 0.082 / 0.174 | 0.081 / 0.158 | 0.082 / 0.177 |
| pełne ROI | 1 | 0.078 / 0.148 | = local | 0.077 / 0.148 | = local |

\* CD-LIN local psuje pojedyncze okno brzegowe (z ≈ 11.95–12.0 m, patrz niżej): mediana po oknach
dla CD-LIN local wynosi 0.129 / 0.095 / 0.087 MPa przy L = 1 / 10 / 25 cm, czyli tyle co AN-JOINT.
AN-JOINT (odwrócenie modelu fizycznego) jest na takie okna odporny, regresja odwrotna nie.

Minimalna długość okna (L i wszystkie dłuższe spełniają kryterium; `min_window_length.md`):

| model / tryb | RMSE_p ≤ 0.1 MPa | RMSE_T ≤ 0.2 °C | oba | ≤ +10 % pełnego ROI | ≤ +20 % |
|---|---|---|---|---|---|
| AN-JOINT local | 25 cm | 1 cm | 25 cm | 50 cm | 25 cm |
| AN-JOINT global | 100 cm | 25 cm | 100 cm | 350 cm | 100 cm |
| CD-LIN local | 25 cm | 25 cm | 25 cm | 50 cm | 25 cm |
| CD-LIN global | 100 cm | 50 cm | 100 cm | 350 cm | 100 cm |

Progi 0.1 MPa / 0.2 °C są przykładowe (klucz `targets` w configu) – do ustalenia z prowadzącym.

### Odpowiedzi na pytania z punktu 7

**Czy lokalna kalibracja jest lepsza od globalnej?** Tak, wyraźnie, dla wszystkich L < 1 m:
przy L = 5 cm błąd ciśnienia spada z 0.40 do 0.12 MPa (3×), przy L = 25 cm z 0.15 do 0.09 MPa.
Przy L = 1 m oba tryby się zrównują. Przyczyną jest niejednorodność czułości (niżej) – jeden zestaw
współczynników nie pasuje do wszystkich fragmentów włókna.

**Czy czułości są przestrzennie jednorodne?** Czułości temperaturowe – praktycznie tak, ciśnieniowe – nie
(`sensitivity_homogeneity.md`, `sensitivity_L5cm.png`, `sensitivity_L25cm.png`). Dla okien 25 cm:

| współczynnik | średnia | CV | rozrzut / niepewność dopasowania |
|---|---|---|---|
| dμ_X/dT (środek zakresu) | −1.443 GHz/°C | 0.45 % | – |
| dμ_Y/dT | −1.428 GHz/°C | 0.44 % | – |
| C_X | 0.718 GHz/MPa | 1.7 % | 7.8 |
| C_Y | −1.933 GHz/MPa | 1.2 % | 11.4 |
| C_d = C_X − C_Y | 2.651 GHz/MPa | 0.64 % | – |

- **C_Y ma systematyczny profil w kształcie litery U**: ok. −1.89 GHz/MPa na obu brzegach ROI
  i −1.97 GHz/MPa w środku (z ≈ 10.5 m). Rozrzut między oknami jest ~11× większy niż niepewność
  dopasowania w oknie, więc to nie jest szum. C_X ma lustrzany, słabszy profil.
- **Dla krótkich okien (5 cm) widać okresowy wzór** w czułości dμ_Y/dT i w błędzie okna, z okresem
  rzędu 10–15 cm. Hipoteza (do potwierdzenia przez prowadzącego): geometria ułożenia włókna w komorze,
  np. nawinięcie. Uwaga: rozrzut czułości to też część błędu, który przy krótkich oknach daje
  różnicę local vs global.
- Wniosek: przy pomiarze rozłożonym warto kalibrować lokalnie (lub przynajmniej uwzględnić profil C(z)).

**Czy występują fragmenty generujące systematycznie większe błędy?** Tak (`outlier_windows.md`, okna
z RMSE > mediana + 3·MAD, `window_rmse_L*cm.png`):
- **brzegi ROI**: z ≈ 8.53–8.55 m i z ≈ 11.95–12.0 m. Na końcu ROI włókno wychodzi z komory –
  sygnał X przechodzi z ok. −29 GHz do 0, `clean_roi` wycina tam większość punktów jako outliery
  (w ostatnich binach kanału X zostaje 0–1 punkt), a resztki przejścia psują okno.
  Rekomendacja: zawęzić ROI do ok. 8.6–11.9 m albo jawnie odrzucać strefy przejścia,
- **pojedyncze okna wewnątrz** (L = 1 cm): z ≈ 8.71–8.73, 10.60 i 11.11 m dla obu modeli
  (czyli cecha włókna / pomiaru, nie modelu), dodatkowo 10.24 i 10.41–10.43 m tylko dla CD-LIN.
  Przy L = 25 cm odstaje pierwsze okno (8.5–8.75 m, środek 8.62 m) – wpływ strefy wejścia do komory.

**Jaka jest minimalna długość odcinka dla założonej dokładności?** Z kalibracją lokalną:
**25 cm** daje RMSE_p ≤ 0.1 MPa i RMSE_T ≤ 0.2 °C (0.09 MPa / 0.16 °C), czyli 14 niezależnych
punktów pomiarowych na 3.5 m zamiast jednego. Pogorszenie względem pełnego ROI: ≤ 20 % od 25 cm,
≤ 10 % od 50 cm. Z kalibracją globalną potrzeba 1 m.

**Kompromis rozdzielczość – dokładność**: przy kalibracji lokalnej RMSE rośnie łagodnie
(od 0.078 MPa dla pełnego ROI do 0.147 MPa dla 1 cm, czyli ~2× przy 350× lepszej rozdzielczości).
To dużo wolniej niż ~1/√L (ok. 19× przy 350× krótszym oknie), czego oczekiwalibyśmy przy szumie
niezależnym punkt do punktu. Sugeruje to, że dominuje błąd wspólny dla całego włókna (powtarzalność
stanu, dokładność etykiet dT i p), a nie szum lokalny – uśrednianie ponad ~25–50 cm niewiele już daje.

### Pliki wynikowe (`Output_files/runs/<data>_spatial_windows/`)

`rmse_vs_L.*`, `rmse_vs_L.png`, `rmse_vs_L_window_median.png`, `min_window_length.*`,
`window_rmse_profiles.csv`, `window_rmse_L{1,5,25}cm.png`, `outlier_windows.*`,
`sensitivity_L{5,25}cm.{csv,png}`, `sensitivity_homogeneity.*`.

### Ograniczenia

- Czyszczenie outlierów (`clean_roi`) jest liczone względem mediany całego ROI – jeżeli fragment
  włókna naprawdę ma inny sygnał (strefa przejścia), zostanie wycięty. Dla analizy przestrzennej
  może być lepsze czyszczenie lokalne (do rozważenia).
- Okna 1 cm ≈ długość bramki OBR (1 cm) – sąsiednie okna nie są w pełni niezależne.
- Niepewności dopasowania czułości (SE) zakładają niezależne powtórzenia; powtórzenia w jednym
  stanie są skorelowane, więc SE jest raczej zaniżone – stosunek rozrzut/SE traktować orientacyjnie.
