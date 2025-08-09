# 🎓 Reprodukcja Eksperymentów Pracy Dyplomowej

**Kompletna instrukcja odtworzenia wszystkich 15 eksperymentów z pracy dyplomowej pt. "Opracowanie metody re-identyfikacji zawodników w nagraniach wideo"**

## 📋 Przegląd

Ten projekt umożliwia pełną reprodukcję wyników z pracy dyplomowej, zawierającej systematyczne porównanie:

- **Uczenie metryki** vs **Podejście klasyfikacyjne**
- **12 eksperymentów uczenia metryki** (pozycje 1-8, 10-12 w rankingu)
- **3 eksperymenty klasyfikacyjne** (pozycje 9, 13-15 w rankingu)
- **Najlepszy wynik:** 54.7% mAP (OSNet + Hard mining)

## 🎯 Wyniki do Reprodukcji (z Tabeli 4.4)

| Poz. | Eksperyment | Architektura | mAP (%) | Rank-1 (%) | Podejście |
|------|-------------|---------------|---------|-------------|-----------|
| 1 | hard_mining_osnet | OSNet | 54.7 | 44.1 | Metryka |
| 2 | optimized_osnet | OSNet | 54.2 | 43.5 | Metryka |
| 3 | sampling_8instances | ResNet50 | 52.9 | 41.4 | Metryka |
| 4 | contrastive_siamese | ResNet50 | 52.7 | 39.3 | Metryka |
| 5 | baseline_resnet50 | ResNet50 | 52.3 | 41.3 | Metryka |
| 6 | margin_05 | ResNet50 | 51.8 | 40.7 | Metryka |
| 7 | advanced_sampling | OSNet | 51.5 | 42.4 | Metryka |
| 8 | margin_01 | ResNet50 | 48.8 | 40.7 | Metryka |
| **9** | **classif_efficientnet_b3** | **EfficientNet-B3** | **48.1** | **37.8** | **Klasyfikacja** |
| 10 | arch_densenet121 | DenseNet121 | 47.4 | 35.1 | Metryka |
| 11 | arch_osnet | OSNet | 46.4 | 34.5 | Metryka |
| 12 | loss_pure_triplet | ResNet50 | 46.4 | 38.4 | Metryka |
| **13** | **classif_efficientnet_b1** | **EfficientNet-B1** | **44.9** | **34.7** | **Klasyfikacja** |
| **14** | **classif_resnet50** | **ResNet50** | **35.9** | **22.9** | **Klasyfikacja** |
| **15** | **default_baseline** | **ResNet50** | **35.6** | **21.3** | **Klasyfikacja** |

## 🚀 Szybki Start

### 1. Wymagania Systemowe

**Minimalne wymagania (zgodne z pracą dyplomową):**
- **GPU:** NVIDIA RTX 4070 TI Super (16GB VRAM) lub podobna
- **RAM:** 16GB (zalecane 32GB)
- **Dysk:** ~35GB wolnego miejsca (18GB dataset + 17GB wyniki)
- **System:** Windows 11 + WSL2 lub Linux
- **Python:** 3.8+
- **CUDA:** 12.4

**Czas wykonania:**
- **Wszystkie 15 eksperymentów:** ~60 godzin (4h × 15)
- **Dataset download:** ~60-90 minut (18GB, jednorazowo)

### 2. Instalacja Środowiska

#### Krok 1: Klonowanie repozytorium
```bash
git clone https://github.com/lrozewicz/dfsagfdgsfdhjkjk.git
cd dfsagfdgsfdhjkjk
```

#### Krok 2: Tworzenie środowiska wirtualnego
```bash
# Windows (Git Bash / WSL2)
python -m venv reid_env
source reid_env/Scripts/activate  # Windows Git Bash
# lub
source reid_env/bin/activate      # WSL2/Linux

# Upgrade pip
python -m pip install --upgrade pip
```

#### Krok 3: Instalacja PyTorch z CUDA
```bash
# CUDA 12.4 (zgodnie z pracą dyplomową)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Alternatywnie dla CUDA 11.8 (jeśli problemy)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Krok 4: Instalacja zależności
```bash
pip install -r requirements.txt
pip install -e .
```

#### Krok 5: Usunięcie potencjalnych konfliktów (opcjonalnie)
```bash
# Jeśli błąd z uprawnieniami
rm -f torchreid/metrics/rank_cylib/rank_cy.cp311-win_amd64.pyd
```

### 3. Weryfikacja Instalacji

```bash
# Test importu torchreid
python -c "import torchreid; print('✅ torchreid imported successfully')"

# Test CUDA
python -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}')"

# Test GPU
python -c "import torch; print(f'✅ GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Szybki test funkcjonalności (~5-10 minut)
python run_quick_test.py
```

**✅ Jeśli wszystkie testy przechodzą, instalacja jest gotowa!**

## 🏃‍♂️ Uruchamianie Eksperymentów

### Opcja A: Wszystkie 15 Eksperymentów (Pełna Reprodukcja)

```bash
# Uruchomienie WSZYSTKICH eksperymentów z pracy dyplomowej
python run_all_experiments.py
```

**Co się dzieje:**
1. **Automatyczny download:** SoccerNet-v3 dataset (~18GB) - jednorazowo
2. **15 eksperymentów:** Uruchamianych sekwencyjnie w kolejności rankingu
3. **Czas wykonania:** ~60 godzin (4h × 15 eksperymentów)
4. **Wyniki:** Zapisywane w `log/` z timestampami

### Opcja B: Wybrane Eksperymenty (Elastyczne)

```bash
# Zobacz dostępne opcje (15 eksperymentów z pracy dyplomowej)
python run_selected_experiments.py --list

# Uruchom TOP-3 eksperymenty z rankingu
python run_selected_experiments.py hard_mining_osnet optimized_osnet sampling_8instances

# Uruchom eksperymenty klasyfikacyjne (pozycje 9, 13-15)
python run_selected_experiments.py classif_efficientnet_b3 classif_efficientnet_b1 classif_resnet50 default_baseline

# Uruchom porównanie architektur
python run_selected_experiments.py baseline_resnet50 arch_densenet121 arch_osnet

# Uruchom badania ablacyjne marginesu
python run_selected_experiments.py margin_01 baseline_resnet50 margin_05

# Najlepszy eksperyment (pozycja 1)
python run_selected_experiments.py hard_mining_osnet

# Podgląd bez uruchamiania
python run_selected_experiments.py hard_mining_osnet optimized_osnet --dry-run
```

### Opcja C: Pojedyncze Eksperymenty

```bash
# Najlepszy wynik z pracy (54.7% mAP)
python benchmarks/baseline/main.py \
    --config-file benchmarks/baseline/configs/hard_mining_experiment.yaml \
    data.save_dir log/hard_mining_reproduction

# Baseline eksperyment (52.3% mAP)
python benchmarks/baseline/main.py \
    --config-file benchmarks/baseline/configs/baseline_60epoch.yaml \
    data.save_dir log/baseline_reproduction

# Najlepszy eksperyment klasyfikacyjny (48.1% mAP)
python benchmarks/baseline/main.py \
    --config-file benchmarks/baseline/configs/classification_efficientnet_b3.yaml \
    data.save_dir log/classification_reproduction
```

## 📊 Dataset SoccerNet-v3

**Automatyczne pobieranie:** Dataset zostanie automatycznie pobrany przy pierwszym uruchomieniu.

**Charakterystyka (zgodnie z pracą dyplomową):**
- **Obrazy treningowe:** 248 234 (1% używane w eksperymentach)
- **Całkowity zbiór:** 376 096 obrazów
- **Tożsamości:** 1 569 zawodników
- **Rozdzielczość:** 256×128 pikseli
- **Źródło:** 400 meczów z 6 głównych lig europejskich

**Struktura:**
```
datasets/soccernetv3/reid/
├── train/              # 248 234 obrazów treningowych
│   └── [identity_id]/  # Katalogi dla każdej tożsamości
├── valid/
│   ├── query/          # Zapytania walidacyjne
│   └── gallery/        # Galeria walidacyjna
├── test/
│   ├── query/          # Zapytania testowe
│   └── gallery/        # Galeria testowa
└── challenge/          # Zbiór wyzwaniowy
    └── gallery/
```

## 🔍 Analiza Wyników

### Lokalizacja Wyników Eksperymentów

Po zakończeniu eksperymentów wyniki znajdziesz w katalogu `log/`. Każdy eksperyment tworzy własny folder z timestampem:

## 📁 Mapowanie Konfiguracji → Katalogi Wyników

| **Konfiguracja YAML** | **Katalog Wyników** | **Opis Eksperymentu** |
|----------------------|---------------------|----------------------|
| `hard_mining_experiment.yaml` | `hard_mining_osnet_[timestamp]/` | Najlepszy wynik (54.7% mAP) |
| `optimized_experiment.yaml` | `optimized_osnet_[timestamp]/` | Zoptymalizowany OSNet (54.2% mAP) |
| `sampling_many_instances.yaml` | `sampling_8instances_[timestamp]/` | 8 instancji na tożsamość (52.9% mAP) |
| `contrastive_loss.yaml` | `contrastive_siamese_[timestamp]/` | Podejście kontrastywne (52.7% mAP) |
| `baseline_60epoch.yaml` | `baseline_resnet50_[timestamp]/` | Baseline ResNet50 (52.3% mAP) |
| `ablation_margin_05.yaml` | `margin_05_[timestamp]/` | Margin 0.5 ablacja (51.8% mAP) |
| `advanced_optimization.yaml` | `advanced_informative_sampling/` | Zaawansowane próbkowanie (51.5% mAP) |
| `ablation_margin_01.yaml` | `margin_01_[timestamp]/` | Margin 0.1 ablacja (48.8% mAP) |
| `classification_efficientnet_b3.yaml` | `classification_efficientnet_b3/` | **Najlepszy klasyfikacyjny (48.1% mAP)** |
| `arch_densenet121.yaml` | `arch_densenet121_[timestamp]/` | DenseNet121 (47.4% mAP) |
| `arch_osnet.yaml` | `arch_osnet_[timestamp]/` | OSNet baseline (46.4% mAP) |
| `loss_pure_triplet.yaml` | `loss_pure_triplet_[timestamp]/` | Czysty triplet loss (46.4% mAP) |
| `classification_efficientnet_b1.yaml` | `classification_efficientnet_b1/` | **EfficientNet-B1 klasyfikacja (44.9% mAP)** |
| `classification_resnet50.yaml` | `classification_resnet50/` | **ResNet50 klasyfikacja (35.9% mAP)** |
| `default_baseline.yaml` | `default_baseline/` | **Framework baseline (35.6% mAP)** |

**Uwaga:** `[timestamp]` to automatycznie generowany znacznik czasu w formacie `YYYYMMDD_HHMMSS` (np. `20250809_143022`)

### Gdzie Znaleźć Konkretne Wyniki

**Dla każdego eksperymentu sprawdź:**

```bash
# Główny plik z logami treningu i wynikami finalnymi
cat log/[nazwa_eksperymentu]/train.log-[data-timestamp]

# Przykład:
cat log/advanced_informative_sampling/train.log-2025-06-06-16-46-34

# Wszystkie zachowane modele
ls log/[nazwa_eksperymentu]/model/

# Wyniki testowe w formacie JSON
cat log/[nazwa_eksperymentu]/ranking_results_soccernetv3_test_*.json
```

**W pliku `train.log-[timestamp]` znajdziesz:**
- Postęp treningu dla każdej epoki
- Wyniki walidacji co 10 epok
- **Finalne wyniki "Evaluating soccernetv3 (source)"** na końcu pliku (najważniejsze dla pracy dyplomowej):
  ```
  => Evaluating soccernetv3 (source)
  ** Results **
  mAP: 54.7%
  CMC curve
  Rank-1  : 44.1%
  Rank-5  : 65.8%
  Rank-10 : 75.2%
  Rank-20 : 82.1%
  ```

### Szczegółowa Analiza Konkretnego Eksperymentu

```bash
# Wyświetl tylko FINALNE wyniki testowe (ostatnie w pliku)
grep -A 15 "Evaluating soccernetv3 (source)" log/hard_mining_osnet_*/train.log-* | tail -15
```

## ⚙️ Konfiguracja Eksperymentów

### Główne Parametry (zgodnie z pracą dyplomową)

**Wspólne ustawienia:**
- **Epoki:** 60
- **Dataset subset:** 1% (training_subset: 0.01)
- **Rozdzielczość:** 256×128
- **Optymalizator:** Adam
- **Ewaluacja:** co 10 epok
- **Augmentacje:** random_flip (podstawowe)

**Różnice między eksperymentami:**
```yaml
# Uczenie metryki (12 eksperymentów)
loss:
  name: triplet  # lub kombinacja triplet + cross-entropy
batch_size: 128  # ResNet50
batch_size: 64   # OSNet, DenseNet121
learning_rate: 0.0003

# Podejście klasyfikacyjne (3 eksperymenty)
engine: classification
loss:
  name: softmax
  label_smooth: true
learning_rate: 0.001  # wyższe niż w uczeniu metryki
batch_size: 32   # EfficientNet-B3
batch_size: 64   # EfficientNet-B1
batch_size: 128  # ResNet50
```

### Modyfikacja Ustawień

```bash
# Zwiększenie danych treningowych (dłuższy trening, lepsze wyniki)
# Edytuj plik .yaml:
soccernetv3:
  training_subset: 0.05  # 5% zamiast 1%

# Zwiększenie liczby epok
train:
  max_epoch: 100  # 100 zamiast 60

# Dostosowanie batch size do GPU
train:
  batch_size: 64  # jeśli problemy z pamięcią
```


## 🤝 Reprodukowalność

Ten framework zapewnia **100% reprodukowalność** eksperymentów z pracy dyplomowej:

- ✅ **Identyczne środowisko:** Windows 11 + WSL2, CUDA 12.4, PyTorch 2.0+
- ✅ **Identyczne dane:** SoccerNet-v3, 1% subset, 248,234 obrazów treningowych
- ✅ **Identyczne parametry:** 60 epok, batch sizes, learning rates
- ✅ **Identyczne architektury:** ResNet50, OSNet, DenseNet121, EfficientNet-B1/B3
- ✅ **Identyczne techniki:** Hard mining, triplet+CE, klasyfikacja

**Wyniki powinny być identyczne (±1% ze względu na randomizację).**