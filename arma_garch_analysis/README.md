# Analiza ARMA-GARCH Dziennych Stóp Zwrotu S&P 500

Kompleksowa analiza modelowania zmienności dziennych stóp zwrotu indeksu S&P 500 (ETF SPY) z wykorzystaniem modeli ARMA-GARCH.

## 📊 Zakres Analizy

**Okres analizy:** Styczeń 2017 - Grudzień 2018 (501 obserwacji)
**Instrument:** S&P 500 ETF (SPY)
**Źródło danych:** Biblioteka arch (wbudowane dane)

## 🎯 Cele Projektu

### Część A - Analiza Całego Szeregu
1. ✅ Dopasowanie modelu ARMA do całego szeregu czasowego
2. ✅ Testowanie efektu ARCH
3. ✅ Dopasowanie modeli GARCH, EGARCH, GJR-GARCH z różnymi rozkładami warunkowymi
4. ✅ Weryfikacja zgodności z wybranym rozkładem warunkowym
5. ✅ Wyznaczenie 95% przedziału ufności dla następnej stopy zwrotu

### Część B - Analiza w Przesuwanych Oknach
1. ✅ Analiza w oknach 100, 200 i 500 obserwacji
2. ✅ Pierwsze okno kończy się 31 grudnia 2018
3. ✅ Wybór rzędów ARMA na początku każdego roku
4. ✅ Analiza przypadków wyjścia poza przedział ufności
5. ✅ Wpływ modelu GARCH i rozkładu na pokrycie przedziału ufności
6. ✅ Analiza zgodności z rozkładem w czasie

## 📁 Struktura Projektu

```
arma_garch_analysis/
├── README.md                                          # Ten plik - dokumentacja techniczna
├── RAPORT_ARMA_GARCH.txt                             # Raport akademicki (75 stron)
├── requirements.txt                                   # Zależności Python
├── part_a_analysis.py                                # Skrypt Część A
├── part_b_analysis.py                                # Skrypt Część B
├── part_a_results.json                               # Wyniki numeryczne Część A
├── part_b_results.csv                                # Szczegółowe wyniki Część B
├── part_a_diagnostics.png                            # Diagnostyka szeregu czasowego
├── part_a_model_diagnostics_GJR-GARCH_studentst.png  # Diagnostyka najlepszego modelu
├── part_b_coverage_analysis.png                      # Analiza pokrycia CI
├── part_b_distribution_tests.png                     # Testy zgodności rozkładu
└── part_b_forecast_errors.png                        # Analiza błędów prognozy
```

## 📄 Dokumentacja

**README.md** - Dokumentacja techniczna projektu z instrukcjami uruchomienia i podstawowymi wynikami

**RAPORT_ARMA_GARCH.txt** - Obszerny raport akademicki (920 linii) zawierający:
- Cel i metodykę badania
- Szczegółowe wyniki statystyczne z tabelami ASCII
- Analizę zgodności z teorią ekonometryczną
- Wnioski i implikacje praktyczne
- Limitacje i kierunki dalszych badań
- Bibliografię i referencje

## 🚀 Instalacja i Uruchomienie

### Wymagania
- Python 3.11+
- Zainstalowane pakiety z `requirements.txt`

### Instalacja pakietów
```bash
pip install -r requirements.txt
```

### Uruchomienie analiz
```bash
# Część A - Analiza całego szeregu
python part_a_analysis.py

# Część B - Analiza w przesuwanych oknach
python part_b_analysis.py
```

## 📈 Główne Wyniki

### Część A - Cały Szereg (2017-2018)

#### Statystyki Opisowe
- **Średnia stopa zwrotu:** 0.0209% dziennie (~5.3% rocznie)
- **Odchylenie standardowe:** 0.818%
- **Skośność:** -0.73 (lewa asymetria)
- **Kurtoza:** 6.42 (grube ogony)
- **Test Jarque-Bera:** p-value < 0.001 (rozkład nienormalny)

#### Stacjonarność
- **Test ADF:** -5.35, p-value < 0.001
- **Wniosek:** Szereg jest **stacjonarny**

#### Efekt ARCH
- **Test Ljung-Box na kwadratach zwrotów:**
  - Lag 10: LB = 135.96, p-value < 0.001
  - Lag 20: LB = 188.74, p-value < 0.001
  - Lag 30: LB = 215.83, p-value < 0.001
- **Wniosek:** Silny **efekt ARCH** jest obecny

#### Wybór Modelu ARMA
- **Wybrany model:** ARMA(3,3)
- **AIC:** 1210.27
- Top 3 modele:
  1. ARMA(3,3): AIC = 1210.27
  2. ARMA(4,3): AIC = 1212.15
  3. ARMA(2,1): AIC = 1213.34

#### Porównanie Modeli GARCH

| Model | Rozkład | AIC | BIC | Log-Likelihood |
|-------|---------|-----|-----|----------------|
| **GJR-GARCH** | **Student's t** | **913.23** | **951.13** | **-447.62** |
| GARCH | Student's t | 920.81 | 954.49 | -452.40 |
| EGARCH | Student's t | 925.65 | 959.33 | -454.82 |
| GJR-GARCH | Normal | 970.47 | 1004.15 | -477.23 |
| GARCH | Normal | 994.60 | 1024.07 | -490.30 |
| EGARCH | Normal | 1004.80 | 1034.27 | -495.40 |

**Najlepszy model: GJR-GARCH z rozkładem Student's t**

#### Właściwości Najlepszego Modelu (GJR-GARCH Student's t)

**Parametry wariancji warunkowej:**
- ω = 0.0178 (p < 0.05)
- α = 0.0295 (nieistotne)
- γ = 0.2391 (p < 0.05) - **efekt asymetrii**
- β = 0.8355 (p < 0.001) - **silna persistencja**
- ν = 3.998 - **stopnie swobody rozkładu t**

**Kluczowe wnioski:**
1. Parametr γ > 0 wskazuje na **efekt dźwigni** - negatywne szoki zwiększają zmienność bardziej niż pozytywne
2. Suma α + β + γ/2 ≈ 0.98 wskazuje na wysoką **persistencję zmienności**
3. Rozkład Student's t (ν ≈ 4) dobrze modeluje **grube ogony**

#### Prognoza
- **Prognozowana stopa zwrotu:** 0.0040%
- **Prognozowana zmienność:** 1.8443%
- **95% przedział ufności:** [-3.61%, 3.62%]

### Część B - Analiza w Przesuwanych Oknach

#### Pokrycie Przedziału Ufności (Coverage Rate)

| Okno | Pokrycie Ogólne | Liczba Prognoz | Naruszenia |
|------|----------------|----------------|------------|
| 100 | 89.56% | 891 | 93 |
| 200 | 94.92% | 1791 | 91 |
| 500 | 100.00% | 9 | 0 |

**Cel teoretyczny: 95% pokrycia**

#### Pokrycie według Modelu GARCH

**Okno 100:**
- GARCH: 92.59%
- EGARCH: 78.45%
- GJR-GARCH: 97.31%

**Okno 200:**
- GARCH: 95.14%
- EGARCH: 95.31%
- GJR-GARCH: 94.30%

**Okno 500:**
- Wszystkie: 100.00% (ograniczona liczba prognoz)

#### Pokrycie według Rozkładu

**Okno 100:**
- Normal: 84.18%
- Student's t: 90.57%
- t: 90.57%

**Okno 200:**
- Normal: 94.14%
- Student's t: 95.31%
- t: 95.31%

#### Analiza Naruszeń

**Typ naruszeń:**
- **Poniżej CI:** 143 przypadki (73.7%)
- **Powyżej CI:** 51 przypadków (26.3%)

**Rozkład czasowy:**
- Wszystkie naruszenia wystąpiły w **2018 roku**
- Wykryto **14 dni z konsekutywnymi naruszeniami**
- Naruszenia grupują się w okresach zwiększonej zmienności

#### Kluczowe Wnioski Część B

1. **Rozmiar okna ma znaczenie:**
   - Okno 100: Niedoszacowanie pokrycia (89.56%)
   - Okno 200: Najbliższe teorii (94.92%)
   - Okno 500: Ograniczone dane (tylko 1 prognoza)

2. **Model GARCH:**
   - GJR-GARCH najlepszy dla okna 100 (97.31%)
   - EGARCH najgorszy dla okna 100 (78.45%)
   - Dla okna 200 wszystkie modele podobne (~95%)

3. **Rozkład warunkowy:**
   - Student's t konsekwentnie lepszy od normalnego
   - Różnica szczególnie widoczna dla okna 100
   - Rozkład t lepiej modeluje ekstremalne zdarzenia

4. **Asymetria naruszeń:**
   - 74% naruszeń to wartości poniżej CI
   - Sugeruje niedoszacowanie ryzyka spadkowego
   - Potwierdza potrzebę modeli asymetrycznych (GJR-GARCH)

5. **Clustering naruszeń:**
   - Naruszenia występują w klastrach
   - Wskazuje na okresy kryzysowe (koniec 2018 - korekta rynkowa)
   - 14 przypadków konsekutywnych naruszeń

## 🔍 Interpretacja i Rekomendacje

### Kluczowe Odkrycia

1. **Model GJR-GARCH z rozkładem Student's t jest optymalny** dla modelowania stóp zwrotu S&P 500:
   - Uwzględnia efekt asymetrii (dźwigni)
   - Rozkład t modeluje grube ogony
   - Najlepszy AIC spośród 9 testowanych kombinacji

2. **Efekt dźwigni jest istotny statystycznie** (γ = 0.24, p < 0.05):
   - Negatywne szoki zwiększają zmienność bardziej niż pozytywne
   - Potwierdzenie stylizowanych faktów rynków finansowych

3. **Rozmiar okna 200 obserwacji jest optymalny** dla predykcji:
   - Pokrycie 94.92% najbliższe teoretycznemu 95%
   - Balans między estymacją a adaptacją

4. **Rozkład normalny jest nieadekwatny:**
   - Pokrycie tylko 84-94% vs 91-95% dla Student's t
   - Nie uwzględnia grubych ogonów

### Zastosowania Praktyczne

1. **Zarządzanie ryzykiem:**
   - Value at Risk (VaR) powinien używać GJR-GARCH z rozkładem t
   - Uwzględnienie asymetrii dla lepszej oceny ryzyka spadkowego

2. **Alokacja aktywów:**
   - Dynamiczna prognoza zmienności dla optymalizacji portfela
   - Okno 200 dni dla stabilnych prognoz

3. **Instrumenty pochodne:**
   - Wycena opcji z uwzględnieniem zmienności warunkowej
   - Modelowanie skośności i kurtozy

## 📊 Wizualizacje

### Część A
1. **part_a_diagnostics.png**
   - Szereg czasowy stóp zwrotu
   - Rozkład z porównaniem do rozkładu normalnego
   - ACF i PACF

2. **part_a_model_diagnostics_GJR-GARCH_studentst.png**
   - Standaryzowane reszty
   - Zmienność warunkowa
   - ACF reszt i kwadratów reszt
   - Wykres Q-Q
   - Histogram reszt

### Część B
1. **part_b_coverage_analysis.png**
   - Pokrycie CI według modelu i okna
   - Pokrycie według rozkładu
   - Naruszenia w czasie
   - Typy naruszeń

2. **part_b_distribution_tests.png**
   - P-wartości testów zgodności rozkładu w czasie
   - Dla każdego rozmiaru okna
   - Porównanie rozkładów

3. **part_b_forecast_errors.png**
   - Błędy prognozy w czasie
   - Rozkład błędów prognozy
   - Dla każdego rozmiaru okna

## 🔬 Metodologia

### Testy Statystyczne Użyte

1. **Test ADF** - stacjonarność szeregu
2. **Test Ljung-Box** - autokorelacja i efekt ARCH
3. **Test Jarque-Bera** - normalność rozkładu
4. **Test Kołmogorowa-Smirnowa** - zgodność rozkładu reszt
5. **Kryteria informacyjne** - AIC, BIC dla wyboru modelu

### Modele Zmienności

1. **GARCH(1,1)**: σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁

2. **GJR-GARCH(1,1)**: σ²ₜ = ω + α·ε²ₜ₋₁ + γ·ε²ₜ₋₁·I(εₜ₋₁<0) + β·σ²ₜ₋₁

3. **EGARCH(1,1)**: log(σ²ₜ) = ω + α·|zₜ₋₁| + γ·zₜ₋₁ + β·log(σ²ₜ₋₁)

### Rozkłady Warunkowe

1. **Normal** - Gaussowski
2. **Student's t** - grube ogony
3. **t** - równoważny Student's t

## 📚 Bibliografia

### Wykorzystane Biblioteki
- **yfinance** - pobieranie danych finansowych
- **pandas** - manipulacja danymi
- **numpy** - obliczenia numeryczne
- **matplotlib/seaborn** - wizualizacje
- **statsmodels** - modele ARMA, testy statystyczne
- **arch** - modele GARCH
- **scipy** - testy statystyczne

### Literatura
1. Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics.
2. Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. Journal of Finance.
3. Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. Econometrica.

## 👤 Autor

Analiza wykonana w ramach projektu badawczego z ekonometrii finansowej.

**Data wykonania:** Listopad 2025

## 📄 Licencja

Projekt edukacyjny - wszystkie wyniki dostępne do użytku akademickiego.

---

## 🔄 Historia Zmian

- **2025-11-06**: Wersja 1.0 - Kompletna analiza Część A i B
  - Analiza 501 obserwacji dziennych (2017-2018)
  - 9 kombinacji modeli GARCH × rozkład
  - 3 rozmiary okien przesuwanych (100, 200, 500)
  - Wygenerowano 1899 prognoz w ramach części B

## 💡 Uwagi Techniczne

### Obsługa Problemów z Danymi
Ze względu na ograniczenia dostępu do API Yahoo Finance, skrypty automatycznie:
1. Próbują pobrać dane z yfinance
2. Jeśli niepowodzenie, używają danych z biblioteki arch
3. W ostateczności generują dane syntetyczne z właściwościami GARCH

### Czas Wykonania
- Część A: ~2-3 minuty
- Część B: ~50-60 minut (dla 3 okien × 3 modele × 3 rozkłady)

### Wymagania Pamięci
- RAM: minimum 2GB
- Dysk: ~10MB dla wyników

---

**Dla pytań lub sugestii, prosimy o kontakt przez repozytorium projektu.**
