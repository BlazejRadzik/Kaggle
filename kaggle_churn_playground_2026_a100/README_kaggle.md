# Kaggle Solution (Playground Churn) — A100 ROC-AUC Ensemble

## Konkurs
- Nazwa: **Predict Customer Churn**
- Platforma: **Kaggle Playground Series**
- Domyślny slug danych (wejścia): `playground-series-s6e3`

## Co robi ten projekt
Ten repozytorium zawiera jeden skrypt treningowy/wnioskowania:
- `colab_churn_ensemble.py`

Skrypt trenuje **zestaw modeli tablicowych** na `train.csv` i generuje predykcje dla `test.csv` jako plik:
- `submission.csv` (blend finalny)

Dodatkowo zapisuje:
- `submission_stack_only.csv` (wariant bez blendu: tylko meta)
- `submission_rank_lgb_cat.csv` (rank-średnia z LightGBM+CatBoost; często bywa stabilniejsze na LB)

## Modele w ensemble
1. **XGBoost** (`XGBClassifier`, early stopping na walidacji fold`)
2. **LightGBM** (`lgb.train`, early stopping na walidacji fold; preferowany GPU, z fallback do CPU)
3. **CatBoost** (`CatBoostClassifier`, log-loss jako metryka do early stopping na GPU)
4. **HistGradientBoostingClassifier** (sklearn; lekki model do poprawy kalibracji/stacku)

## Strategie łączenia (ROC-AUC)
- Trening jest prowadzony w schemacie **Stratified K-Fold**.
- Powstają predykcje **OOF** (out-of-fold) dla każdego modelu.
- Następnie jest:
  - **logit stack** (regresja logistyczna) na wektorach OOF
  - oraz **dynamiczny blend**: siatka wag mieszająca:
    - stack meta
    - liniowo ważone modele (na podstawie AUC modeli na OOF)
    - rank-średnią predykcji modeli (stabilizacja do LB)

## Parametry i czas (ważne przy A100)
Skrypt ma tryb jakości/czasu przez zmienną środowiskową:

`CHURN_PRESET`:
- `balanced` (domyślnie): kompromis czas vs stabilność (zwykle najlepszy do submitów)
  - ok. `N_SPLITS=8` i 3 seed-sety
- `fast`: szybszy bieg (mniej foldów, większa wariancja)
- `max`: najbliżej “maks jakości” (więcej foldów/seeds, dłużej)

Przykład:
```python
import os
os.environ["CHURN_PRESET"] = "balanced"  # fast | max
```

## Wejście/wyjście (w Kaggle/Colab)
Skrypt oczekuje, że w katalogu roboczym istnieją:
- `train.csv`
- `test.csv`

W Kaggle notebook przed uruchomieniem trzeba skopiować pliki `train.csv` i `test.csv`
z `../input/...` do `/kaggle/working/` (czyli do bieżącego katalogu roboczego notebooka).

## Uruchomienie (Kaggle Notebook — komórki do wklejenia)
### Komórka 1: setup bibliotek
```python
!pip -q install pandas numpy scikit-learn lightgbm xgboost catboost
```

### Komórka 2: skopiuj train/test do katalogu roboczego
```python
import glob, shutil

base = "/kaggle/input"
train_candidates = glob.glob(base + "/*/train.csv")
test_candidates  = glob.glob(base + "/*/test.csv")

assert len(train_candidates) > 0 and len(test_candidates) > 0

shutil.copy(train_candidates[0], "train.csv")
shutil.copy(test_candidates[0], "test.csv")
```

### Komórka 3: uruchom kod
Wklej treść `colab_churn_ensemble.py` do tej komórki albo uruchom przez `exec(...)`.
Najprościej: wgraj plik `colab_churn_ensemble.py` do notebooka i:
```python
%run colab_churn_ensemble.py
```

## Wynik z Twojego A100 run (zrzut)
- OOF ROC-AUC (przybliżone wartości z loga):
  - XGB: `0.914459`
  - LGB: `0.916703`
  - CAT: `0.916017`
  - HGB: `0.915926`
  - STACK(meta): `0.916642`
  - BLEND(Oof): `0.916432`
- LB (Kaggle submission):
  - `submission.csv`: **0.91385**

## Dlaczego działa szybko na A100
- XGBoost i LightGBM używają GPU (z fallbackiem do CPU).
- CatBoost ma GPU ustawione przez `task_type="GPU"` gdy dostępne.
- Do ograniczenia kosztu pamięci:
  - rzutowanie preprocessingów do `float32` gdy to `numpy` (mniej RAM, szybsze kopie).

## Pliki w repo
- `colab_churn_ensemble.py` — pełny pipeline end-to-end (train + pred + zapis submission)
- `README_kaggle.md` — opis rozwiązania

