# Pricing Trasporti

Pipeline end-to-end per costruire un modello di pricing trasporti:
EDA, preprocessing, attualizzazione, matching ordini/quotazioni, benchmark modelli e training finale.

## Pipeline

1. `01_eda.py`
2. `02_preprocessing.py`
3. `03_attualizzazione.py`
4. `04_matching.py`
5. `05_benchmark.py`
6. `06_training.py`

## Input richiesti

Per compatibilita con il codice attuale, i file vanno messi nella root del progetto con nomi esatti.

- `01_risultati_ordini.xlsx` (input base)
- `TavoleStream.xlsx` (solo se `METODO="tavole"` in `03_attualizzazione.py`)

Output intermedi/finali vengono generati automaticamente (`02_preprocessed.xlsx`, `03_attualizzato.xlsx`, ecc.).

## Schema dati consigliato (input base)

Colonne core consigliate in `01_risultati_ordini.xlsx`:

- identificativi: `idordine`, `idquotazione`
- target/raw price: `importotrasp` (o `importo`)
- geometria/volume: `km_tratta`, `peso_totale`, `altezza`, `lunghezza_max`, `misure`
- date: `data_ordine` (opzionale: `data_carico`, `data_scarico`)
- categoriali: `tipo_carico`, `naz_carico`, `naz_scarico`
- coordinate (opzionali ma utili): `latitudine_carico`, `longitudine_carico`, `latitudine_scarico`, `longitudine_scarico`

Il codice gestisce diverse colonne opzionali e fallback (vedi script `02_*`, `03_*`, `05_*`, `06_*`).

## Setup

```bash
pip install -r requirements.txt
```

## Esecuzione

```bash
python 01_eda.py
python 02_preprocessing.py
python 03_attualizzazione.py
python 04_matching.py
python 05_benchmark.py
python 06_training.py
```

## Nota privacy

I file Excel originali e i risultati derivati sono stati rimossi/anonimizzati dalla repo.
Questa versione contiene codice e istruzioni di caricamento.
## Research Profile

- Research keywords: transport pricing, feature engineering, inflation adjustment, matching, predictive modeling.
- Positioning: applied operations-research and ML project for pricing systems.
- Open-source status: this repository is open source and intended for reproducible research and education.

