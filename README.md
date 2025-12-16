# MLOps Assignment 02

A complete MLOps project demonstrating version control (Git + DVC), CI/CD (GitHub Actions), containerization (Docker), workflow orchestration (Airflow), API development (FastAPI), and cloud deployment (AWS EC2 + S3).

## Project Structure
```
MLOps_Ass02/
├── src/                    # Source code
│   └── train.py           # Model training script
├── api/                   # API code
│   └── main.py           # FastAPI application
├── data/                  # Data directory (DVC tracked)
│   └── dataset.csv       # Dataset
├── models/               # Trained models
│   └── model.pkl        # Saved model
├── tests/               # Unit tests
│   └── test_train.py   # Training tests
├── airflow/            # Airflow DAGs
│   └── train_pipeline.py
├── .github/           # GitHub Actions
│   └── workflows/
│       └── ci.yml
├── dvcstore/         # DVC local remote
├── .gitignore
├── requirements.txt
├── Dockerfile
├── dvc.yaml         # DVC pipeline
└── README.md
```

## Setup Instructions

### 1. Clone and Setup
```bash
git clone <https://github.com/ShoaibHaider113/mlops-assignment-02>
cd MLOps_Ass02
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. DVC Setup
```bash
dvc pull  # Pull data from DVC remote
```

### 3. Run Training
```bash
python src/train.py
```

### 4. Run API
```bash
uvicorn api.main:app --reload
```

### 5. Run Tests
```bash
pytest tests/
```

## Docker Usage
```bash
# Build image
docker build -t mlops-app .

# Run training
docker run mlops-app

# Run API
docker run -p 8000:8000 mlops-api
```

## AWS Deployment
- **S3 Bucket**: [Your bucket URL]
- **EC2 Public Endpoint**: [Your EC2 public IP]:8000

## Links
- **GitHub Repository**: [https://github.com/ShoaibHaider113/mlops-assignment-02]
- **Docker Hub**: [(https://hub.docker.com/repositories/shab8)]

## Author
[Muhammad Shoaib Haider]
