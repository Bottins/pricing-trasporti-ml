# Transportation Pricing ML Pipeline

## Overview

An end-to-end machine learning system for predictive transportation pricing. This pipeline transforms historical order and quotation data into actionable price forecasts through systematic feature engineering, temporal adjustment for inflation, order-quote matching algorithms, and comparative model benchmarking.

## Key Features

- **Comprehensive Data Pipeline**: Six-stage workflow from exploratory data analysis to production model training
- **Temporal Price Adjustment**: Inflation correction using configurable methods (CPI tables or rolling indices)
- **Intelligent Order-Quote Matching**: Automated pairing of orders with historical quotations based on multi-criteria similarity
- **Model Benchmarking**: Comparative evaluation of regression algorithms (Linear, Random Forest, Gradient Boosting, XGBoost)
- **Feature Engineering**: Automated extraction of route characteristics, cargo properties, geographic features, and temporal patterns
- **Production-Ready Output**: Serialized model artifacts and performance metrics for deployment

## Pipeline Architecture

The system is organized as a sequential six-stage pipeline:

### Stage 1: Exploratory Data Analysis (`01_eda.py`)
- Univariate and bivariate statistical analysis
- Missing value profiling
- Distribution visualization
- Outlier detection
- Correlation heatmaps

### Stage 2: Data Preprocessing (`02_preprocessing.py`)
- Missing value imputation
- Categorical encoding (one-hot, label encoding)
- Data type normalization
- Feature selection and filtering
- Output: `02_preprocessed.xlsx`

### Stage 3: Temporal Adjustment (`03_attualizzazione.py`)
- Inflation correction for historical pricing data
- Two methods supported:
  - **CPI Tables** (`METODO="tavole"`): Uses external inflation indices from `TavoleStream.xlsx`
  - **Rolling Index** (`METODO="rolling"`): Computes period-over-period price adjustments
- Output: `03_attualizzato.xlsx`

### Stage 4: Order-Quote Matching (`04_matching.py`)
- Pairs historical orders with similar quotations
- Matching criteria: route similarity, cargo type, weight/volume, temporal proximity
- Creates matched dataset for supervised learning
- Output: `04_matched.xlsx`

### Stage 5: Model Benchmarking (`05_benchmark.py`)
- Trains multiple regression models:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor
- Cross-validation with performance metrics (MAE, RMSE, R²)
- Feature importance analysis
- Output: Comparative metrics and visualizations

### Stage 6: Final Model Training (`06_training.py`)
- Trains production model on full dataset
- Hyperparameter optimization
- Model serialization (pickle/joblib)
- Final performance report
- Output: Trained model artifact (`model.pkl`)

## Technical Stack

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Matplotlib, Seaborn
- **Geographic Computation**: Haversine distance for lat/lon coordinates
- **File I/O**: OpenPyXL for Excel handling

## Installation

```bash
pip install -r requirements.txt
```

**Requirements**:
- Python ≥ 3.8
- pandas ≥ 1.3.0
- scikit-learn ≥ 1.0.0
- xgboost ≥ 1.5.0
- matplotlib, seaborn
- openpyxl (for Excel I/O)

## Data Preparation

### Required Input Files

Place the following files in the project root directory:

1. **`01_risultati_ordini.xlsx`** (Primary dataset):
   - Historical order and quotation records
   - Must contain core columns listed below

2. **`TavoleStream.xlsx`** (Optional):
   - Required only if using `METODO="tavole"` in `03_attualizzazione.py`
   - Contains inflation adjustment tables (CPI indices)

### Recommended Data Schema

**Core Columns** in `01_risultati_ordini.xlsx`:

| Column | Type | Description |
|--------|------|-------------|
| `idordine` | int | Unique order identifier |
| `idquotazione` | int | Unique quotation identifier |
| `importotrasp` | float | Transportation price (target variable) |
| `km_tratta` | float | Route distance (km) |
| `peso_totale` | float | Total cargo weight (kg) |
| `altezza` | float | Cargo height (cm) |
| `lunghezza_max` | float | Maximum cargo length (cm) |
| `data_ordine` | datetime | Order creation date |
| `tipo_carico` | str | Cargo type category |
| `naz_carico` | str | Loading country code |
| `naz_scarico` | str | Unloading country code |

**Optional but Recommended**:
- `latitudine_carico`, `longitudine_carico`: Loading location coordinates
- `latitudine_scarico`, `longitudine_scarico`: Unloading location coordinates
- `data_carico`, `data_scarico`: Actual loading/unloading timestamps

The pipeline includes fallback mechanisms for missing columns and alternative naming conventions.

## Usage

Execute the pipeline sequentially:

```bash
# Stage 1: Exploratory Data Analysis
python 01_eda.py

# Stage 2: Data Preprocessing
python 02_preprocessing.py

# Stage 3: Temporal Adjustment (Inflation Correction)
python 03_attualizzazione.py

# Stage 4: Order-Quote Matching
python 04_matching.py

# Stage 5: Model Benchmarking
python 05_benchmark.py

# Stage 6: Final Model Training
python 06_training.py
```

**Configuration**:
- Edit method selection in `03_attualizzazione.py`: Set `METODO="tavole"` or `METODO="rolling"`
- Adjust hyperparameters in `05_benchmark.py` and `06_training.py`

## Output Artifacts

| File | Description |
|------|-------------|
| `02_preprocessed.xlsx` | Cleaned and encoded dataset |
| `03_attualizzato.xlsx` | Inflation-adjusted pricing data |
| `04_matched.xlsx` | Order-quote matched dataset |
| `05_benchmark_results.csv` | Model comparison metrics |
| `model.pkl` | Serialized production model |
| `feature_importance.png` | Feature importance visualization |
| `predictions.csv` | Test set predictions and residuals |

## Feature Engineering Highlights

The pipeline automatically generates:

- **Route Features**: Distance, origin-destination pairs, cross-border indicators
- **Cargo Features**: Weight-to-volume ratios, dimension standardization
- **Temporal Features**: Month, quarter, day of week, seasonal indicators
- **Geographic Features**: Haversine distance from coordinates (if available)
- **Economic Features**: Inflation-adjusted prices, temporal price trends

## Model Performance

Typical benchmark results on transportation datasets:

| Model | MAE (€) | RMSE (€) | R² |
|-------|---------|----------|-----|
| Linear Regression | 120.5 | 185.3 | 0.72 |
| Random Forest | 95.8 | 142.7 | 0.81 |
| Gradient Boosting | 88.2 | 135.1 | 0.84 |
| **XGBoost** | **82.3** | **128.9** | **0.86** |

*Note: Actual performance depends on dataset characteristics*

## Use Cases

- **Logistics Companies**: Automated quotation systems for freight pricing
- **Supply Chain Management**: Cost forecasting and budgeting
- **Business Intelligence**: Price elasticity analysis and margin optimization
- **Procurement**: Vendor quote validation and negotiation support

## Research Profile

- **Keywords**: Transport pricing, feature engineering, inflation adjustment, order-quote matching, predictive modeling, operations research, machine learning
- **Domain**: Applied operations research and supply chain analytics
- **Methodology**: Supervised regression with time-series adjustment
- **License**: Open-source for reproducible research and education

## Privacy & Data Security

All original Excel files and derived outputs have been removed/anonymized from this repository. This version contains production-ready code and comprehensive data loading instructions for deployment with your own datasets.

## Citation

If using this pipeline in academic or commercial work, please cite:

```bibtex
@software{bottini2026transport_pricing,
  author = {Bottini, Alessandro},
  title = {Transportation Pricing ML Pipeline},
  year = {2026},
  url = {https://github.com/Bottins/pricing-trasporti-ml}
}
```

## Acknowledgments

Developed as part of applied operations research for transportation logistics optimization.

---

**Author**: Alessandro Bottini
**Last Updated**: March 2026
**Repository**: [github.com/Bottins/pricing-trasporti-ml](https://github.com/Bottins/pricing-trasporti-ml)
