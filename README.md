# Cloud Telemetry Intelligence Platform

An end-to-end machine learning system for ingesting cloud and network telemetry, detecting anomalous behavior, predicting performance regressions, and serving low-latency inference through a production-style API.

## Why this project

This project is designed to mirror a real infrastructure ML workflow rather than a demo-only model notebook. It focuses on:

- telemetry ingestion at scale
- data cleaning and feature engineering
- supervised and unsupervised ML
- regression and classification
- evaluation with operational metrics
- API deployment and reproducible experiments

The goal is to build something that looks credible for cloud, networking, observability, and ML engineering roles.

## Core use cases

The platform solves three related problems from the same telemetry stream:

1. Classification: determine whether a recent telemetry window contains an anomaly.
2. Regression: predict near-future latency, throughput degradation, or failure rate.
3. Unsupervised learning: detect novel or weakly labeled failure patterns with anomaly detection and clustering.

## Example input data

The system is intended to ingest telemetry such as:

- CPU utilization
- memory usage
- network latency
- packet drops
- request throughput
- error rates
- service health events
- structured log summaries

Input data can come from:

- public telemetry or anomaly datasets
- synthetic telemetry generated with controlled fault injection
- exported CSV, JSON, Parquet, or SQL-backed records

## Planned architecture

```mermaid
flowchart LR
    A[Raw Telemetry Sources] --> B[Ingestion Pipeline]
    B --> C[Cleaning and Validation]
    C --> D[Feature Engineering and Windowing]
    D --> E[Training Datasets]
    E --> F1[Classification Models]
    E --> F2[Regression Models]
    E --> F3[Unsupervised Models]
    F1 --> G[Evaluation]
    F2 --> G
    F3 --> G
    G --> H[Model Registry and Artifacts]
    H --> I[FastAPI Inference Service]
    I --> J[Dashboard and Batch Reports]
```

## Target tech stack

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- PyTorch or TensorFlow for optional neural baselines
- FastAPI
- PostgreSQL
- Docker
- Matplotlib
- Jupyter
- pytest
- GitHub Actions

## Proposed repository structure

```text
cloud-telemetry-intelligence-platform/
  README.md
  data/
    raw/
    processed/
    synthetic/
  notebooks/
    01_exploration.ipynb
    02_feature_engineering.ipynb
    03_model_comparison.ipynb
    04_error_analysis.ipynb
  src/
    ingestion/
    features/
    models/
    training/
    evaluation/
    serving/
    db/
    utils/
  tests/
    ingestion/
    features/
    models/
    serving/
  docker/
  scripts/
  configs/
```

## Detailed build plan

### Phase 1: Ingestion layer

Build a reproducible ingestion pipeline that accepts metrics and event data from CSV, JSON, or SQL sources.

Required work:

- create schemas for metrics, events, and labels
- validate timestamps, service identifiers, and host identifiers
- support batch ingestion from local files
- store raw and cleaned data separately
- add data quality checks for null spikes, duplicate rows, and invalid ranges

Implementation notes:

- Use `pandas` for initial file loaders.
- Use `SQLAlchemy` or direct PostgreSQL connectors for persistence.
- Make each ingestion job idempotent so reruns do not duplicate records.

### Phase 2: Cleaning and preprocessing

Prepare telemetry for model training.

Required work:

- impute or drop missing values depending on feature criticality
- standardize units and timestamp precision
- resample metrics into fixed windows such as 1 minute or 5 minute intervals
- normalize numerical features
- create rolling statistics such as mean, std, max, min, slope, and change rate
- join event summaries with metric windows
- generate labels for anomaly classification and regression targets

Key engineered features:

- rolling latency mean and variance
- request error ratio
- packet drop burst count
- CPU to throughput imbalance
- short-term versus long-term drift
- event frequency by component

### Phase 3: Model development

Train and compare multiple model families.

Classification models:

- logistic regression baseline
- random forest classifier
- XGBoost classifier

Regression models:

- linear regression baseline
- random forest regressor
- XGBoost regressor

Unsupervised models:

- Isolation Forest
- KMeans for failure pattern clustering

Optional advanced model:

- sequence autoencoder or LSTM only if the dataset supports it and there is enough time for proper tuning

### Phase 4: Evaluation

Measure both predictive quality and operational usefulness.

Classification metrics:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- confusion matrix

Regression metrics:

- RMSE
- MAE
- MAPE where appropriate

Operational metrics:

- inference latency
- batch scoring throughput
- false-positive rate under noisy load
- stability across services and time ranges

Analysis outputs:

- feature importance plots
- model comparison tables
- threshold sweeps
- error analysis notebooks
- ablations on feature groups

### Phase 5: Serving and deployment

Expose a lightweight inference API for real-time or batch scoring.

Serving requirements:

- FastAPI endpoint for anomaly classification
- FastAPI endpoint for regression prediction
- batch scoring endpoint for offline evaluation
- persisted prediction records in PostgreSQL
- Dockerized local deployment

Recommended API shape:

```text
POST /predict/anomaly
POST /predict/regression
POST /predict/batch
GET  /health
GET  /metrics
```

### Phase 6: Reliability and developer workflow

Make the project feel like production-grade engineering rather than a notebook dump.

Required engineering work:

- unit tests for ingestion, feature transforms, and inference
- configuration files for experiments
- reproducible training scripts
- linting and formatting
- CI checks through GitHub Actions
- clear separation of raw, processed, and model artifact data

## How to build this project

The recommended implementation order is below.

### Step 1: Create the Python environment

Use Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install the expected dependencies once `requirements.txt` or `pyproject.toml` is added:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost fastapi uvicorn sqlalchemy psycopg[binary] jupyter pytest
```

Optional deep learning extras:

```bash
pip install torch
# or
pip install tensorflow
```

### Step 2: Stand up local services

Run PostgreSQL locally, either through Docker or a local install.

Example with Docker:

```bash
docker run --name telemetry-postgres \
  -e POSTGRES_USER=telemetry \
  -e POSTGRES_PASSWORD=telemetry \
  -e POSTGRES_DB=telemetry_ml \
  -p 5432:5432 \
  -d postgres:16
```

### Step 3: Prepare or generate telemetry data

Start with one of these paths:

- import a public dataset and convert it to a unified schema
- generate synthetic telemetry with injected faults such as latency spikes, packet loss, and throughput collapse

Minimum dataset fields:

- timestamp
- service_name
- host_id
- cpu_pct
- memory_pct
- latency_ms
- throughput_rps
- error_rate
- packet_drop_pct
- event_summary
- anomaly_label

### Step 4: Build ingestion jobs

Implement loaders that:

- read raw files from `data/raw/`
- validate schema and ranges
- write cleaned tables to PostgreSQL
- export curated training data to `data/processed/`

### Step 5: Build feature pipelines

Implement reusable transforms for:

- rolling windows
- lag features
- rate-of-change features
- normalization
- label generation

Persist processed features so training and serving use the same logic.

### Step 6: Train baselines first

Start with simple models before deep learning:

- logistic regression for anomaly classification
- linear regression for next-step latency
- Isolation Forest for unsupervised anomaly detection

Then add tree-based models and compare quality against the baselines.

### Step 7: Evaluate and document results

Create notebooks and scripts that:

- compare model performance
- visualize false positives and misses
- analyze feature importance
- measure inference latency
- summarize tradeoffs between supervised and unsupervised methods

### Step 8: Add a serving layer

Train, serialize, and load the selected models in FastAPI. Add request validation and basic observability.

Run the service locally:

```bash
uvicorn src.serving.api:app --reload
```

### Step 9: Add tests and CI

At minimum, add:

- schema validation tests
- feature pipeline tests
- model smoke tests
- API contract tests

Then configure GitHub Actions to run linting, tests, and basic build checks on every pull request.

## Recommended milestones

1. Telemetry schema, ingestion pipeline, and data validation
2. Feature engineering and baseline notebooks
3. Classification, regression, and unsupervised model training
4. Evaluation dashboard and experiment comparison
5. FastAPI inference service and PostgreSQL integration
6. Docker packaging, tests, and CI

## What success looks like

This project is successful if it can:

- ingest realistic telemetry data reproducibly
- detect anomalous windows with strong precision and recall
- predict performance degradation with reasonable error bounds
- compare supervised and unsupervised methods credibly
- serve predictions through a clean API
- demonstrate strong ML and software engineering practices in one repository

## Resume-ready summary

Built an end-to-end ML platform for cloud and network telemetry that ingests, cleans, and models time-series infrastructure data for anomaly detection and performance regression prediction using Python, pandas, scikit-learn, and FastAPI.

