# Drone Detection MLOps

Pipeline MLOps pour la détection de drones par YOLOv8, avec tracking MLflow, orchestration Airflow, serving FastAPI et déploiement Kubernetes.

**Dataset** : [YOLO Drone Detection Dataset](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset) (1012 images train, 347 images valid, 1 classe "drone").

## Stack technique

| Composant | Outil | Justification |
|---|---|---|
| ML Framework | YOLOv8n (Ultralytics) | Format natif du dataset, léger, rapide |
| Orchestrateur | Apache Airflow 2.7.3 | Imposé par le sujet |
| Tracking & Registry | MLflow | Imposé par le sujet |
| Stockage objets | MinIO | S3-compatible, local, gratuit |
| Base de données | PostgreSQL 15 | Backend store pour Airflow et MLflow |
| API Serving | FastAPI | Swagger auto, async, léger |
| WebApp | Gradio 5 | Upload d'images + visualisation bboxes simple |
| CI/CD | GitHub Actions | Imposé par le sujet |
| Conteneurisation | Docker / Docker Compose | Imposé par le sujet |
| Gestion dépendances | Poetry | Lock file cross-platform, groupes dev/prod |
| Déploiement | Kubernetes (Minikube) | Imposé par le sujet |
| Monitoring | Prometheus + Grafana | Métriques API temps réel, dashboards |

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌───────────┐
│  Kaggle API ├────►│    Airflow       ├────►│  MinIO    │
│  (dataset)  │     │  (3 DAGs)       │     │  (S3)     │
└─────────────┘     └───────┬──────────┘     └─────┬─────┘
                            │                      │
                    ┌───────▼──────────┐     ┌─────▼─────┐
                    │  YOLOv8n         │     │  MLflow   │
                    │  (entraînement)  ├────►│  (tracking│
                    └───────┬──────────┘     │  +registry│
                            │                └─────┬─────┘
                    ┌───────▼──────────┐           │
                    │  FastAPI         │◄──────────┘
                    │  POST /predict   │   (charge le modèle)
                    └───────┬──────────┘
                            │
                    ┌───────▼──────────┐
                    │  Gradio          │
                    │  (interface web) │
                    └──────────────────┘
```

## Prérequis

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)
- Docker & Docker Compose
- Compte [Kaggle](https://www.kaggle.com/) (token API)

## Guide de démarrage

### Etape 1 : Installation

```bash
poetry install
```

### Etape 2 : Téléchargement du dataset

Placer le token Kaggle dans `~/.kaggle/kaggle.json` puis :

```bash
poetry run python -m src.data.download
```

Le dataset (1012 images train, 347 images valid, 1 classe "drone") est extrait dans `data/drone_dataset/`.

### Etape 3 : Lancer l'environnement dev

```bash
docker compose up -d
```

| Service | URL | Identifiants |
|---|---|---|
| Airflow | http://localhost:8080 | admin / admin |
| MLflow | http://localhost:5001 | - |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| API Swagger | http://localhost:8000/docs | - |
| WebApp Gradio | http://localhost:7860 | - |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin / admin |

### Etape 4 : Entraînement

En local (5 epochs, ~12 min sur CPU M4) :

```bash
poetry run python -c "from src.training.train import train; train(epochs=5, imgsz=640, batch=8)"
```

Ou via Airflow : déclencher le DAG `train_pipeline`.

Le modèle est automatiquement :
- Loggé dans MLflow (métriques + artefacts)
- Enregistré dans le Model Registry (`drone-detection-yolo`)
- Stocké sur MinIO (bucket `mlflow`)

### Etape 5 : Tester l'API

```bash
curl -X POST http://localhost:8000/predict -F "file=@data/drone_dataset/valid/images/pic_001.jpg"
```

Ou via Swagger : http://localhost:8000/docs

### Etape 6 : WebApp

Ouvrir http://localhost:7860, uploader une image, ajuster le seuil de confiance.

### Etape 7 : Tests

```bash
poetry run pytest tests/unit/ -v         # 4 tests (validation labels/images)
poetry run pytest tests/integration/ -v  # 2 tests (API health + predict)
```

### Etape 8 : Déploiement Kubernetes

```bash
docker build -f docker/Dockerfile.api -t drone-detection-api .
docker build -f docker/Dockerfile.webapp -t drone-detection-webapp .
minikube start
kubectl apply -f k8s/
```

## Structure du projet

```
.
├── .github/workflows/
│   ├── ci.yml                  # Lint (ruff) + tests unitaires + intégration
│   └── cd.yml                  # Build images Docker + push DockerHub + deploy K8s
├── airflow/dags/
│   ├── data_ingestion.py       # DAG : download Kaggle + validation
│   ├── train_pipeline.py       # DAG : entraînement YOLOv8 + log MLflow
│   └── continuous_training.py  # DAG : réentraînement hebdo + comparaison + promotion
├── src/
│   ├── data/
│   │   ├── download.py         # Téléchargement dataset via Kaggle CLI
│   │   └── preprocess.py       # Validation images (PIL) et labels (format YOLO)
│   ├── training/
│   │   ├── train.py            # Entraînement YOLOv8n + logging MLflow + register model
│   │   └── evaluate.py         # Évaluation mAP + comparaison prod vs nouveau modèle
│   ├── serving/
│   │   └── api.py              # FastAPI : /health, /predict (upload image → détections)
│   └── webapp/
│       └── app.py              # Gradio : upload image → image annotée avec bboxes
├── tests/
│   ├── unit/
│   │   └── test_preprocess.py  # 4 tests : validation labels valides/invalides, image corrompue
│   └── integration/
│       └── test_api.py         # 2 tests : health check, prédiction sur image dummy
├── k8s/
│   ├── api-deployment.yml      # Deployment + Service (LoadBalancer :8000)
│   ├── webapp-deployment.yml   # Deployment + Service (LoadBalancer :7860)
│   └── monitoring.yml          # Prometheus + Grafana (Deployments + Services)
├── docker/
│   ├── Dockerfile.api          # python:3.10-slim + libGL + poetry deps + serving code
│   ├── Dockerfile.webapp       # python:3.10-slim + gradio + requests + pillow
│   ├── Dockerfile.mlflow       # python:3.10-slim + mlflow + psycopg2 + boto3
│   ├── Dockerfile.training     # python:3.10-slim + poetry deps + training code
│   └── init-db.sh              # Crée la base "mlflow" séparée dans PostgreSQL
├── data/
│   ├── data.yaml               # Config YOLO : chemins train/val, nc=1, names=[drone]
│   └── drone_dataset/          # Dataset extrait (gitignored)
├── monitoring/
│   ├── prometheus/prometheus.yml  # Config Prometheus (scrape API /metrics)
│   └── grafana/provisioning/     # Datasources + dashboard Grafana pré-provisionné
├── docker-compose.yml          # 9 services pour l'environnement dev
├── pyproject.toml              # Dépendances Poetry (main + dev)
└── .gitignore
```

## Services Docker Compose

| Service | Image | Port | Rôle |
|---|---|---|---|
| postgres | postgres:15 | 5432 | BDD Airflow (`airflow`) + BDD MLflow (`mlflow`) |
| minio | minio/minio | 9000, 9001 | Stockage S3 : buckets `drone-detection` et `mlflow` |
| minio-init | minio/mc | - | Init : création des buckets au démarrage |
| mlflow | custom | 5001→5000 | Tracking server + Model Registry, backend PostgreSQL, artefacts sur MinIO |
| airflow-init | apache/airflow:2.7.3 | - | Init : `db upgrade` + création user admin |
| airflow-webserver | apache/airflow:2.7.3 | 8080 | Interface web Airflow |
| airflow-scheduler | apache/airflow:2.7.3 | - | Exécution des DAGs |
| api | custom | 8000 | API FastAPI de serving |
| webapp | custom | 7860 | Interface Gradio |
| prometheus | prom/prometheus | 9090 | Collecte des métriques API (scrape /metrics) |
| grafana | grafana/grafana | 3000 | Dashboards de monitoring (admin/admin) |

### Choix d'architecture notables

- **Bases PostgreSQL séparées** : Airflow et MLflow utilisent le même serveur PostgreSQL mais des bases différentes (`airflow` et `mlflow`) pour éviter les conflits de migrations Alembic.
- **MLflow `--allowed-hosts "*"`** : nécessaire pour que l'interface web soit accessible depuis le navigateur (Docker expose via `0.0.0.0`).
- **Webapp isolée** : le Dockerfile webapp installe uniquement gradio/requests/pillow via pip (pas poetry) pour éviter les conflits de versions avec FastAPI.

## Composants détaillés

### Ingestion des données (`src/data/`)

- **`download.py`** : télécharge le dataset via `kaggle datasets download`, décompresse dans `data/drone_dataset/`.
- **`preprocess.py`** : valide chaque image (PIL.verify) et chaque label (format YOLO : `class x_center y_center width height`, valeurs dans [0,1]). Supprime les paires invalides.

### Entraînement (`src/training/`)

- **`train.py`** :
  - Désactive le callback MLflow intégré d'Ultralytics (`yolo_settings.update({"mlflow": False})`) pour contrôler manuellement le logging.
  - Configure les credentials S3/MinIO via variables d'environnement.
  - Entraîne YOLOv8n, log les hyperparamètres et métriques dans MLflow.
  - Upload `best.pt` comme artefact dans MinIO via MLflow.
  - Enregistre le modèle dans le Model Registry (`drone-detection-yolo`).

- **`evaluate.py`** : récupère les métriques du modèle en production depuis MLflow, compare avec un nouveau modèle sur `mAP50`.

### DAGs Airflow (`airflow/dags/`)

| DAG | Trigger | Tâches |
|---|---|---|
| `data_ingestion` | Manuel | download_dataset → preprocess_data |
| `train_pipeline` | Manuel | train_model (entraînement + log MLflow + register) |
| `continuous_training` | Hebdomadaire | retrain → compare_models → promote_model OU skip_promotion |

Le DAG `continuous_training` utilise un `BranchPythonOperator` pour promouvoir le nouveau modèle en "Production" uniquement s'il bat l'ancien sur mAP50.

### API Serving (`src/serving/api.py`)

- **GET `/health`** : retourne `{"status": "ok"}`.
- **POST `/predict`** : accepte un fichier image + paramètre `confidence` (défaut 0.25). Retourne les détections avec classe, confiance et bounding box.
- **GET `/metrics`** : expose les métriques Prometheus (predictions_total, errors, latency, detections).

Chargement du modèle (par ordre de priorité) :
1. Chemin local (`LOCAL_MODEL_PATH`)
2. Model Registry MLflow (dernière version Production ou None)
3. Fallback : modèle YOLOv8n pré-entraîné (COCO)

### WebApp (`src/webapp/app.py`)

Interface Gradio avec :
- Input : upload d'image + slider de confiance (0.1 - 1.0)
- Output : image annotée avec bounding boxes rouges et labels
- Communique avec l'API FastAPI via `requests.post`.

### CI/CD (`.github/workflows/`)

**CI (`ci.yml`)** - déclenché sur push main/dev et PR sur main :
1. Lint avec `ruff`
2. Tests unitaires (`tests/unit/`)
3. Tests d'intégration (`tests/integration/`)

**CD (`cd.yml`)** - déclenché sur push main :
1. Build et push des images Docker (API + Webapp) sur DockerHub
2. Déploiement sur Kubernetes

### Kubernetes (`k8s/`)

- **api-deployment.yml** : 1 replica, port 8000, `readinessProbe` sur `/health`, Service LoadBalancer.
- **webapp-deployment.yml** : 1 replica, port 7860, Service LoadBalancer. Variable `API_URL` pointe vers le Service API interne.

## Tests

| Suite | Fichier | Tests | Description |
|---|---|---|---|
| Unit | `test_preprocess.py` | 4 | Validation label valide, format invalide, valeurs hors range, image corrompue |
| Integration | `test_api.py` | 2 | Health check (GET /health), prédiction sur image dummy (POST /predict) |

## Métriques obtenues

Entraînement YOLOv8n, 5 epochs, batch=8, imgsz=640, CPU Apple M4 (~12 min) :

| Métrique | Valeur |
|---|---|
| mAP@0.5 | **0.815** |
| mAP@0.5:0.95 | 0.458 |
| Precision | 0.789 |
| Recall | 0.772 |

## Variables d'environnement

| Variable | Défaut | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5001` | URL du serveur MLflow |
| `MLFLOW_S3_ENDPOINT_URL` | `http://localhost:9000` | URL MinIO pour les artefacts |
| `AWS_ACCESS_KEY_ID` | `minioadmin` | Credentials MinIO |
| `AWS_SECRET_ACCESS_KEY` | `minioadmin` | Credentials MinIO |
| `MODEL_NAME` | `drone-detection-yolo` | Nom du modèle dans le registry |
| `DATA_YAML` | `data/data.yaml` | Chemin vers la config YOLO |
| `LOCAL_MODEL_PATH` | - | Chemin local vers un .pt (prioritaire sur MLflow) |
| `API_URL` | `http://localhost:8000` | URL de l'API (utilisé par la webapp) |

## Logging et Monitoring

### Logging

Tous les modules utilisent le module `logging` Python avec des niveaux adaptés :
- **INFO** : opérations normales (téléchargement, entraînement, prédictions, chargement modèle)
- **WARNING** : fichiers invalides supprimés lors du preprocessing
- **ERROR** : échecs de connexion MLflow, erreurs de prédiction

### Monitoring (Prometheus + Grafana)

L'API expose un endpoint **`GET /metrics`** avec 4 métriques Prometheus :

| Métrique | Type | Description |
|---|---|---|
| `predictions_total` | Counter | Nombre total de prédictions |
| `predictions_errors_total` | Counter | Nombre d'erreurs de prédiction |
| `detections_total` | Counter | Nombre total de drones détectés |
| `prediction_latency_seconds` | Histogram | Latence des prédictions (p50, p95, p99) |

**Prometheus** (port 9090) scrape l'API toutes les 15 secondes.

**Grafana** (port 3000, admin/admin) inclut un dashboard pré-provisionné avec :
- Taux de requêtes (req/s)
- Latence p50/p95/p99
- Taux d'erreur
- Moyenne de détections par requête

## Compatibilité

- **macOS Apple Silicon (M4)** : testé, fonctionne nativement.
- **Windows 10/11** : utiliser Docker Desktop avec backend WSL 2. Les Dockerfiles utilisent `python:3.10-slim` (multi-arch) compatible amd64 et arm64.
- Les chemins dans `data.yaml` utilisent des chemins relatifs POSIX, compatibles partout via YOLO.
