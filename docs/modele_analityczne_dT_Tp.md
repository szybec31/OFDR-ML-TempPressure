# Modele analityczne, etykieta temperatury (dT vs Tp) i cztery schematy walidacji

Dokument opisuje trzy modele analityczne (AN-BL, AN-BL-NEAREST, AN-JOINT), wyjaśnia, dlaczego wyniki dla
etykiety `dT` i `Tp` tak bardzo się różnią, oraz analizuje wyniki w czterech schematach walidacji.

Wszystkie liczby pochodzą z jednego uruchomienia, które można odtworzyć poleceniem:

```
python main.py experiment configs/analytical_dT_vs_Tp.json
```

Wyniki zapisano w `Output_files/runs/20261018_132138_analytical_dT_vs_Tp/`, osobny podkatalog na każdy wariant:
`condition_dT`, `condition_Tp`, `temperature_dT`, …, `block_Tp`.

W uruchomieniu wzięły udział trzy modele analityczne i dwa modele ML jako punkt odniesienia: MO-LR (regresja
liniowa na 8 cechach) oraz CD-LIN (regresja liniowa na `m` i `d`). Modele ML nie mają siatek hiperparametrów,
więc całość liczy się w około 30 sekund.

Oznaczenia: RMSE_p w MPa, RMSE_T w °C. O ile nie napisano inaczej, wartości są średnimi z foldów.

---

## 1. Wspólny model fizyczny

Wszystkie trzy modele analityczne korzystają z tego samego równania z artykułu (Bednarek i in., 2026).
Każdy kanał polaryzacyjny k ∈ {X, Y} reaguje na temperaturę i ciśnienie tak:

```
μ_k = A_k · T² + B_k · T + C_k · p
```

- `μ_k` to średnie przesunięcie widmowe w ROI [GHz],
- `T` to temperatura liczona względem stanu referencyjnego (dla celu `dT`: T = dT, dla celu `Tp`: T = Tp − 23 °C),
- `A_k`, `B_k` opisują (lekko kwadratową) czułość temperaturową, a `C_k` liniową czułość ciśnieniową.

Równanie nie ma wyrazu wolnego, bo w stanie referencyjnym (T = 0, p = 0) przesunięcie wynosi 0.
Pliki OBR zawierają przesunięcie względem jednego globalnego pomiaru referencyjnego, więc warunek ten jest spełniony.

Współczynniki dopasowane do naszych danych (cel `dT`):

| | A [GHz/°C²] | B [GHz/°C] | C [GHz/MPa] |
|---|---|---|---|
| kanał X | −0,0009 | −1,40 | +0,71 |
| kanał Y | −0,0007 | −1,40 | −1,98 |

Temperatura przesuwa oba kanały prawie identycznie (B_X ≈ B_Y), a ciśnienie w przeciwne strony.
Dlatego z pary (μ_X, μ_Y) da się rozdzielić T i p.

### Odwracanie modelu (predykcja)

Mając zmierzone μ_X i μ_Y, model wyznacza T i p w dwóch krokach (`invert_quadratic` w
[physical_model.py](../baselines/models/physical_model.py)):

1. **Eliminacja p.** Po podzieleniu równań przez C_X i C_Y i odjęciu ich ciśnienie znika:
   `μ_X/C_X − μ_Y/C_Y = a·T² + b·T`, gdzie `a = A_X/C_X − A_Y/C_Y`, `b = B_X/C_X − B_Y/C_Y`.
   To równanie kwadratowe względem T.
2. **Wybór pierwiastka i podstawienie.** Z dwóch pierwiastków wybierany jest bliższy środkowi zakresu
   temperatur treningowych. Gdy pierwiastków rzeczywistych brak, brany jest wierzchołek paraboli. Następnie
   `p = (μ_X − A_X·T² − B_X·T) / C_X`.

Trzy modele różnią się wyłącznie tym, **na jakich danych i w jakiej kolejności wyznaczane są A, B i C**.

---

## 2. Trzy modele analityczne

### AN-BL – kalibracja sekwencyjna jak w artykule

Klasa `PhysicalModel(pressure_calibration="strict")`. Kalibracja przebiega w dwóch osobnych krokach:

1. **A_k, B_k** z rekordów `is_temp_calibration`: pomiary przy p = 0 (`zero_start`) ze wszystkich
   temperatur w treningu. Fit `μ_k = A_k·T² + B_k·T` (MNK bez wyrazu wolnego). Wymaga co najmniej 2 poziomów temperatury.
2. **C_k** z rekordów `is_pressure_calibration`: wszystkie ciśnienia z serii referencyjnej dT = 0.
   Fit `μ_k = C_k·p`.

Model **nie ma żadnych wartości domyślnych**. Jeżeli w treningu brakuje którejś grupy rekordów, fold dostaje
status `not_identifiable` (N/A). Dzieje się to w dwóch sytuacjach:
- test zawiera serię dT = 0 (nie ma czym wyznaczyć C),
- test zawiera wszystkie rekordy p = 0 (nie ma czym wyznaczyć A, B).

AN-BL używa tylko ok. 15 % danych treningowych (83 rekordy temperaturowe i 93 ciśnieniowe z 1166).
Pozostałe pomiary, czyli wszystkie kombinacje T > 0 i p > 0, nie biorą udziału w kalibracji.

### AN-BL-NEAREST – AN-BL z zapasową kalibracją ciśnieniową

Klasa `PhysicalModel(pressure_calibration="nearest_temperature")`. Gdy seria dT = 0 jest w treningu, model
jest **identyczny z AN-BL**, co widać w tabelach: te same liczby we wszystkich foldach poza jednym.

Gdy dT = 0 jest w teście, C_k wyznaczane jest z pomiarów ciśnieniowych w **najniższej dostępnej temperaturze
treningowej**. Wcześniej od sygnału odejmowana jest część temperaturowa wyliczona z A_k, B_k:
`μ_k − (A_k·T² + B_k·T) = C_k·p`.

Jedyny fold, w którym ten mechanizm działa, to leave-one-temperature-out z testem dT = 0. C liczone jest wtedy
z serii dT = 16,2 °C:

| | C_X | C_Y |
|---|---|---|
| AN-BL (z dT = 0) | 0,708 | −1,977 |
| AN-BL-NEAREST (z dT = 16,2) | 0,753 | −1,858 |

W tym foldzie AN-BL-NEAREST ma błąd 0,183 MPa / 0,285 °C, a AN-BL jest N/A. Seria 16,2 °C jest przy tym
nietypowa (patrz punkt 3), co psuje tę zastępczą kalibrację.

### AN-JOINT – ta sama fizyka, kalibracja łączna

Klasa `JointPhysicalModel`. Wszystkie współczynniki są dopasowywane **jednocześnie** metodą najmniejszych
kwadratów na wszystkich rekordach treningowych (`is_joint_regression`), w których występują razem
temperatura i ciśnienie:

```
μ_k = A_k·T² + B_k·T + C_k·p     (jeden fit na kanał, dane: wszystkie T i wszystkie p)
```

Nie potrzebuje osobnych serii p = 0 ani dT = 0. Wystarczą 3 stany (T, p), dla których macierz
[T², T, p] ma pełny rząd, więc **nigdy nie jest N/A** w naszych schematach walidacji.

### Porównanie współczynników

Wszystkie wartości dla celu `dT`; AN-JOINT dopasowany na całych danych.

| | A_X | B_X | C_X | A_Y | B_Y | C_Y |
|---|---|---|---|---|---|---|
| AN-BL | −0,0009 | −1,401 | 0,708 | −0,0007 | −1,397 | **−1,977** |
| AN-JOINT | −0,0008 | −1,408 | 0,718 | −0,0008 | −1,393 | **−1,933** |

Część temperaturowa jest praktycznie taka sama. Różni się czułość ciśnieniowa C, przede wszystkim C_Y.

### Dlaczego AN-JOINT jest wyraźnie lepszy dla ciśnienia

Czułość ciśnieniowa **zmienia się z temperaturą** (czułość skrośna). Dopasowanie `μ ~ C·p` osobno w każdej serii daje:

| dT [°C] | 0 | 16,2 | 20,5 | 24,7 | 27,4 | 33 | 37,5 | 41,8 | 45,7 |
|---|---|---|---|---|---|---|---|---|---|
| C_Y [GHz/MPa] | −1,972 | −1,995 | −1,932 | −1,930 | −1,892 | −1,934 | −1,916 | −1,938 | −1,956 |
| C_d = C_X − C_Y | 2,693 | 2,714 | 2,661 | 2,647 | 2,621 | 2,640 | 2,639 | 2,636 | 2,647 |

AN-BL bierze C wyłącznie z serii dT = 0, gdzie różnicowa czułość ciśnieniowa jest o ok. 2 % większa niż
w pozostałych temperaturach. W wyższych temperaturach model przecenia więc czułość i **systematycznie zaniża
ciśnienie**, tym bardziej, im wyższe p. Średni błąd ze znakiem (bias) AN-BL w leave-one-condition-out:

| p [MPa] | 0 | 0,5 | 1 | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| bias AN-BL | −0,03 | −0,01 | −0,13 | −0,18 | −0,09 | −0,09 | −0,16 | −0,19 |
| bias AN-JOINT | 0,02 | 0,05 | −0,07 | −0,11 | 0,02 | 0,04 | −0,01 | −0,02 |

AN-JOINT uśrednia C po wszystkich temperaturach i jego bias jest bliski zera. W każdym schemacie walidacji
AN-BL ma bias ciśnienia od −0,10 do −0,13 MPa, a AN-JOINT od −0,015 do +0,004 MPa.

Dla temperatury różnica między modelami jest mała, bo A i B w obu wariantach wyznacza się z pomiarów
we wszystkich temperaturach.

Ujemny bias przy 1–3 MPa występuje we wszystkich modelach, również w AN-JOINT i MO-LR. Wskazuje to na wspólną
cechę tych pomiarów (np. dokładność zadanego ciśnienia albo histerezę), a nie na błąd konkretnego modelu.

**Wniosek:** słabość AN-BL wynika z **kalibracji sekwencyjnej**, a nie z samego modelu fizycznego. Ta sama
fizyka skalibrowana łącznie (AN-JOINT) osiąga dokładność ciśnienia na poziomie najlepszych modeli ML.
Naturalnym rozszerzeniem byłaby czułość ciśnieniowa zależna od temperatury (C_k = C_k0 + C_k1·T),
co odpowiada modelom PHYS-RES / CD-POLY z listy zadań.

---

## 3. Etykieta temperatury: dT czy Tp

### Skąd biorą się obie etykiety

Nazwa folderu z danymi koduje obie wartości, np. `DT24.7_T50_20` oznacza:
- `dT` = 24,7 – zmiana temperatury (`deltaT_label`),
- `Tp` = 50 – temperatura odczytana z płyty grzewczej (`T_plate_label`),
- `20` – numer serii.

Instrukcja projektu każe modelować `dT` i traktować `Tp` jako metadane. Artykuł rozróżnia temperaturę zadaną
przez płytę od rzeczywistej zmiany temperatury włókna. Cel można przełączyć kluczem
`"temperature_target": "dT" | "Tp"` w dowolnym configu (opis w README).

### Co pokazują dane

Zestawienie per seria. `m(p=0)` to średnia składowa wspólna (μ_X + μ_Y)/2 przy p = 0, czyli sygnał niosący
temperaturę.

| seria | data pomiaru | dT | Tp | Tp − 23 | dT / (Tp − 23) | m(p=0) [GHz] | m / dT | m / (Tp − 23) |
|---|---|---|---|---|---|---|---|---|
| 15 | 18.08.2025 | 0 | 23 | 0 | – | −0,03 | – | – |
| 2 | 04.06.2025 | 16,2 | 42 | 19 | 0,853 | −22,20 | −1,371 | −1,169 |
| 16 | 19.08.2025 | 27,4 | 55 | 32 | 0,856 | −39,03 | −1,424 | −1,220 |
| 19 | 28.08.2025 | 20,5 | 45 | 22 | 0,932 | −29,08 | −1,419 | −1,322 |
| 20 | 29.08.2025 | 24,7 | 50 | 27 | 0,915 | −35,07 | −1,420 | −1,299 |
| 21 | 01.09.2025 | 33,0 | 60 | 37 | 0,892 | −47,20 | −1,430 | −1,276 |
| 25 | 09.09.2025 | 37,5 | 65 | 42 | 0,893 | −53,25 | −1,420 | −1,268 |
| 26 | 10.09.2025 | 41,8 | 70 | 47 | 0,889 | −59,98 | −1,435 | −1,276 |
| 27 | 11.09.2025 | 45,7 | 75 | 52 | 0,879 | −65,59 | −1,435 | −1,261 |

Wnioski z tabeli:
- **Sygnał jest prawie idealnie proporcjonalny do dT**: m/dT ≈ −1,42 GHz/°C we wszystkich seriach poza serią 2.
- **Względem Tp proporcja „pływa”** od −1,17 do −1,32 GHz/°C, a stosunek dT/(Tp − 23) zmienia się
  niemonotonicznie od 0,853 do 0,932.
- **Stosunek dT/(Tp − 23) grupuje się według sesji pomiarowych**:
  - 0,853–0,856 dla serii 2 i 16,
  - 0,915–0,932 dla serii 19–20 (28–29 sierpnia),
  - 0,879–0,893 dla serii 21–27 (wrzesień).

  Każda temperatura była mierzona innego dnia, więc przełożenie „temperatura płyty → temperatura włókna”
  zmienia się między dniami (np. przez inną temperaturę otoczenia lub inny kontakt termiczny).
- **Seria 2 (dT = 16,2) jest nietypowa także względem dT**: m/dT = −1,371 zamiast −1,42. Zmierzono ją
  2,5 miesiąca przed resztą i brakuje w niej pomiarów 0,1–0,5 MPa. W każdym schemacie walidacji to jeden
  z najtrudniejszych foldów.

### „Podłoga” błędu: jak dobrze sygnał może w ogóle opisać etykietę

Najlepsze możliwe dopasowanie etykiety funkcją sygnału (wielomian 2. stopnia w m i d, na wszystkich danych,
bez walidacji) to dolna granica błędu każdego modelu:

| etykieta | RMSE dopasowania | średni błąd na poziom temperatury |
|---|---|---|
| dT | 0,12 °C | od −0,12 do +0,28 °C (największy: seria 2) |
| Tp | **0,74 °C** | od −0,96 do **+1,30 °C** (Tp = 42: +1,22; Tp = 45: −0,96; Tp = 55: +1,30) |

Dla Tp każdy poziom temperatury ma własne przesunięcie, rzędu ±1 °C, którego **żadna funkcja zmierzonego
sygnału nie może wyjaśnić**. Sygnał „wie”, jaka jest temperatura włókna, a nie płyty.

### Wyniki: dT vs Tp

RMSE temperatury dla obu etykiet:

| schemat | AN-BL | AN-BL-NEAREST | AN-JOINT | MO-LR | CD-LIN |
|---|---|---|---|---|---|
| leave-one-condition-out, dT | 0,102 | 0,102 | 0,097 | 0,107 | 0,130 |
| leave-one-condition-out, Tp | 0,586 | 0,586 | 0,614 | 0,610 | 0,614 |
| leave-one-temperature-out, dT | 0,140* | 0,157 | 0,144 | 0,196 | 0,202 |
| leave-one-temperature-out, Tp | 0,831* | 0,873 | 0,815 | 0,851 | 0,758 |
| leave-one-pressure-out, dT | 0,131* | 0,131* | 0,120 | 0,126 | 0,158 |
| leave-one-pressure-out, Tp | 0,754* | 0,754* | 0,739 | 0,707 | 0,740 |
| blok T×p, dT | 0,117 | 0,117 | 0,119 | 0,123 | 0,156 |
| blok T×p, Tp | 0,618 | 0,618 | 0,657 | 0,646 | 0,652 |

\* mniej foldów, bo model jest N/A w jednym foldzie (8/9 lub 13/14).

**RMSE ciśnienia jest identyczne dla obu etykiet**, np. AN-BL 0,113 / 0,113, AN-JOINT 0,061 / 0,061,
MO-LR 0,057 / 0,057 MPa w leave-one-condition-out. Dzieje się tak z dwóch powodów:
- W modelach liniowych (MO-LR, CD-LIN) wyjście ciśnienia to osobna regresja, która w ogóle nie widzi etykiety temperatury.
- W modelach analitycznych ciśnienie wynika głównie z różnicy kanałów. Ponieważ B_X ≈ B_Y, temperatura się
  w niej znosi, a sposób jej opisania prawie nie wpływa na wyznaczone p.

### Co z tego wynika

1. **Przy Tp błąd temperatury wynika z etykiety, nie z modelu.** Wszystkie modele, fizyczne i ML, lądują
   przy 0,6–0,9 °C, czyli przy „podłodze” 0,74 °C. Różnice między modelami (rzędu 0,03 °C) są wtedy bez
   znaczenia, a ranking modeli dla temperatury traci sens.
2. **dT jest zgodna z sensorem prawie idealnie** (m/dT ≈ const). Możliwe są dwa wyjaśnienia:
   - dT to rzeczywista zmiana temperatury włókna, zmierzona niezależnie (np. termoparą przy włóknie).
     Wtedy wyniki dla dT są wiarygodne.
   - dT wyliczono **z samego przesunięcia widmowego** przy założonej czułości. Wtedy estymacja dT jest
     częściowo kołowa (model uczy się odtwarzać przelicznik, którym wyznaczono etykietę), a błędy
     temperatury rzędu 0,1 °C są zbyt optymistyczne.

   Na podstawie samych danych nie da się tego rozstrzygnąć. To jest dokładnie zadanie 2 z listy prowadzącego.
   Trzeba zapytać, jak wyznaczono `deltaT_label`.
3. **Czułość temperaturowa zależy od wyboru etykiety**: B ≈ −1,40 GHz/°C względem dT i ≈ −1,28 GHz/°C
   względem Tp (AN-JOINT). Różnica (~9 %) odpowiada stosunkowi dT/(Tp − 23) ≈ 0,85–0,93, czyli
   włókno nagrzewa się średnio o ok. 90 % tego, co wskazuje płyta.
4. **Ciśnienie można raportować niezależnie od sporu o etykietę temperatury**, bo jego wyniki są identyczne
   dla obu celów.
5. **Rekomendacja:** dT zostaje celem głównym (zgodnie z instrukcją), wyniki dla Tp raportujemy jako test
   kontrolny. Jeśli wyjaśni się, że dT pochodzi z przesunięcia widmowego, błędy temperatury trzeba
   opisywać jako zgodność z przelicznikiem, a nie jako dokładność pomiaru temperatury.

---

## 4. Cztery schematy walidacji

| schemat | co jest w teście jednego folda | foldów | co sprawdza |
|---|---|---|---|
| leave-one-condition-out (`run`, główny) | wszystkie powtórzenia jednej kombinacji (T, p) | 119 | interpolację: ta sama T i to samo p są w treningu w innych kombinacjach |
| leave-one-temperature-out (`run_temp`) | cały poziom temperatury, czyli cała seria | 9 | uogólnianie na niewidzianą temperaturę **i** niewidzianą sesję pomiarową |
| leave-one-pressure-out (`run_pressure`) | jeden poziom ciśnienia we wszystkich temperaturach | 14 | uogólnianie na niewidziane ciśnienie |
| blok T×p (`run_block`) | prostokątny fragment przestrzeni, np. ΔT ≥ 37,5 i p ≥ 7 | 4 | wypełnianie „dziury” w przestrzeni kalibracji |

**Leave-one-series-out (`run_series`) daje te same foldy co leave-one-temperature-out**, bo każda seria to
dokładnie jedna temperatura i jeden dzień pomiarów. Pipeline wypisuje o tym ostrzeżenie przy uruchomieniu.

### Wyniki (cel dT)

| schemat | AN-BL | AN-BL-NEAREST | AN-JOINT | MO-LR | CD-LIN |
|---|---|---|---|---|---|
| condition (119 foldów) | 0,113 / 0,102 | 0,113 / 0,102 | 0,061 / 0,097 | 0,057 / 0,107 | 0,060 / 0,130 |
| temperature (9) | 0,142 / 0,140 (8) | 0,147 / 0,157 | 0,082 / 0,144 | 0,077 / 0,196 | 0,082 / 0,202 |
| pressure (14) | 0,125 / 0,131 (13) | 0,125 / 0,131 (13) | 0,075 / 0,120 | 0,071 / 0,126 | 0,074 / 0,158 |
| blok T×p (4) | 0,144 / 0,117 | 0,144 / 0,117 | 0,068 / 0,119 | 0,061 / 0,123 | 0,065 / 0,156 |

Format: RMSE_p [MPa] / RMSE_T [°C]; w nawiasie liczba foldów, gdy model był w części N/A.

Na 8 foldach wspólnych w leave-one-temperature-out (bez dT = 0) wyniki wyglądają inaczej:

| model | RMSE_p [MPa] | RMSE_T [°C] |
|---|---|---|
| AN-BL / AN-BL-NEAREST | 0,142 | 0,140 |
| AN-JOINT | 0,078 | 0,148 |
| MO-LR | 0,075 | 0,149 |
| CD-LIN | 0,077 | 0,148 |

Jak widać, różnicę w temperaturze robi prawie wyłącznie fold dT = 0 (opis niżej).

### Dlaczego wyniki różnią się między schematami

**1. AN-BL prawie nie zależy od schematu, ale zmienia się zbiór testowy.**
Treningu AN-BL usunięcie jednej kombinacji, jednego ciśnienia > 0 ani jednego bloku praktycznie nie
zmienia, bo kalibruje się tylko na p = 0 i dT = 0. Mimo to jego RMSE_p waha się od 0,113 do 0,144.
Różnicę robi **skład testu**, bo bias AN-BL rośnie z ciśnieniem i zależy od temperatury:
- condition: każda ze 119 kombinacji waży tyle samo, w tym wiele kombinacji niskiego ciśnienia o małym biasie, stąd 0,113.
- blok: dwa z czterech bloków to wysokie ciśnienie (p ≥ 7), gdzie bias jest największy:
  high_T_high_p 0,201 i low_T_high_p 0,180 MPa. Średnia rośnie do 0,144.
- temperature: każdy fold to wszystkie ciśnienia jednej temperatury, a seria 16,2 °C ma bias −0,22 MPa, stąd 0,142.

Wniosek ogólny: **średnich z różnych schematów nie porównuje się bezpośrednio**. Porównuje się modele
w obrębie jednego schematu, na tych samych foldach.

**2. Modele uczone na wszystkich danych tracą dokładność proporcjonalnie do ekstrapolacji.**
AN-JOINT: 0,061 (condition) → 0,068 (blok) → 0,075 (pressure) → 0,082 MPa (temperature).
- **condition** to czysta interpolacja. W treningu są pozostałe ciśnienia tej samej serii, więc model
  „zna” nawet dzień pomiaru. To najłatwiejszy test i najbardziej optymistyczne liczby.
- **blok** usuwa fragment przestrzeni, ale zależność od T i p jest gładka (prawie liniowa w p,
  lekko kwadratowa w T), więc AN-JOINT i MO-LR dobrze wypełniają lukę: 0,027–0,086 MPa w każdym bloku.
- **pressure** wymaga przewidzenia niewidzianego ciśnienia, a na brzegach (p = 0 i p = 10) ekstrapolacji.
  Sygnał jest liniowy w p, więc nadal działa dobrze: bias przy p = 10 MPa wynosi tylko −0,02 MPa dla AN-JOINT.
- **temperature** jest najtrudniejszy. Test to nieznana temperatura i jednocześnie nieznany dzień pomiaru
  z jego własnymi przesunięciami, a fold dT = 0 to ekstrapolacja poza zakres treningowy.

**3. Fold dT = 0 rozstrzyga o temperaturze w leave-one-temperature-out.**

| model | RMSE_T w foldzie dT = 0 [°C] |
|---|---|
| AN-JOINT | **0,112** |
| AN-BL-NEAREST | 0,285 |
| MO-LR | 0,567 |
| CD-LIN | 0,633 |
| AN-BL | N/A |

Najbliższy poziom treningowy to 16,2 °C, więc trzeba ekstrapolować o 16 °C. Model fizyczny wie, że przy
T = 0 i p = 0 sygnał wynosi 0 (brak wyrazu wolnego) i zna kształt krzywej, więc ekstrapoluje poprawnie.
Regresje liniowe z wyrazem wolnym tego ograniczenia nie mają. To jeden z najmocniejszych argumentów za
modelami z fizyczną strukturą.

AN-BL-NEAREST traci, bo jego zastępcza kalibracja C pochodzi z nietypowej serii 16,2 °C.

**4. Seria 16,2 °C jest najtrudniejsza w każdym schemacie.**
W leave-one-temperature-out ma najwyższe RMSE_T dla wszystkich modeli (0,30–0,41 °C) i najwyższe RMSE_p
dla AN-BL (0,253 MPa). Powody: inna sesja pomiarowa (czerwiec), odstająca czułość (m/dT = −1,371) i brak
niskich ciśnień.

**5. AN-BL jest N/A tam, gdzie test zabiera jego dane kalibracyjne.**
- **temperature, fold dT = 0**: brak serii do kalibracji C. AN-BL-NEAREST to obchodzi, AN-JOINT nie ma tego problemu.
- **pressure, fold p = 0**: brak wszystkich rekordów p = 0, więc nie da się wyznaczyć A, B. Tu N/A są oba
  warianty AN-BL; AN-JOINT działa (0,054 MPa bias, 0,116 °C).

Liczba N/A nie jest słabością pipeline'u. To realna cecha kalibracji sekwencyjnej: wymaga dwóch
specjalnych serii pomiarowych, a kalibracja łączna nie.

**6. Dla Tp kolejność schematów jest podobna, ale poziom błędu wyznacza etykieta.**
RMSE_T rośnie od 0,59 (condition) do 0,83 °C (temperature). W leave-one-temperature-out przesunięcie
etykiety danego poziomu (±1 °C) jest całkowicie nieznane, bo cały poziom jest w teście. W condition inne
kombinacje tego samego poziomu są w treningu, ale modele są gładkimi funkcjami sygnału i nie mogą
„zapamiętać” osobnego przesunięcia dla każdego poziomu, więc błąd zostaje blisko podłogi 0,74 °C.

---

## 5. Podsumowanie

1. **Trzy modele analityczne mają tę samą fizykę i różnią się tylko kalibracją.** AN-BL i AN-BL-NEAREST są
   identyczne, gdy dostępna jest seria dT = 0. AN-JOINT kalibruje wszystko naraz i jest od nich wyraźnie
   lepszy w ciśnieniu (≈ 0,06–0,08 zamiast 0,11–0,14 MPa), bo nie przenosi czułości ciśnieniowej z dT = 0
   na wszystkie temperatury.
2. **Kalibracja sekwencyjna ma dwie wady**: systematycznie zaniża ciśnienie (bias −0,10…−0,13 MPa, rosnący z p)
   i nie działa bez specjalnych serii dT = 0 i p = 0.
3. **W temperaturze modele analityczne są na poziomie ML**, a przy ekstrapolacji (fold dT = 0) wyraźnie lepsze.
4. **Wyniki dla Tp są 5–6 razy gorsze dla wszystkich modeli**, bo etykieta płyty ma przesunięcia ±1 °C
   zależne od sesji pomiarowej, niewidoczne w sygnale. Ciśnienie nie zależy od wyboru etykiety.
5. **Do wyjaśnienia z prowadzącym:** jak wyznaczono `deltaT_label`. Od tego zależy, czy błędy temperatury
   ~0,1 °C są rzeczywistą dokładnością, czy zgodnością z przelicznikiem.
6. **Schematy walidacji odpowiadają na różne pytania i dają różne liczby.** Condition jest najbardziej
   optymistyczny (interpolacja), temperature najbardziej wymagający (nowa temperatura i nowy dzień).
   Do raportu: porównywać modele w obrębie schematu i pokazywać kilka schematów obok siebie.
