# Telco Customer Churn Prediction (End-to-End MLOps)

This repository demonstrates a complete, production-grade Machine Learning Operations (MLOps) pipeline. It aims to predict customer churn in the telecommunications sector using historical customer data. 

The architecture bridges the gap between raw data and a highly optimized, served model, employing industry-standard tools like **FastAPI, ONNX, DagsHub, MLflow, and Docker**.

---

## 🏗️ Architecture & Component Deep Dive

The project is structured into distinct modular components, managed by a central configuration system to ensure reproducibility and maintainability.

### 1. Data Ingestion (`data_ingestion.py`)
This component acts as the entry point of the pipeline. It programmatically connects to Kaggle using the `kagglehub` library to fetch the `blastchar/telco-customer-churn` dataset. 
- **Why?** Automating ingestion ensures that the pipeline can run completely unattended, a critical requirement for CI/CD environments. The data is cached locally in the `artifacts/data_ingestion` directory to prevent redundant downloads on subsequent runs.

### 2. Data Validation (`data_validation.py`)
Before any machine learning takes place, the incoming dataset is validated against a strict schema defined in `config/schema.yaml`. 
- **Why?** Data drift or upstream schema changes are common causes of silent ML failures. This component acts as a gatekeeper; if the columns or expected data types change, the pipeline halts immediately, logging the failure status to `artifacts/data_validation/status.txt`.

### 3. Data Transformation (`data_transformation.py`)
This is where raw data is converted into a format suitable for machine learning algorithms. 
- It utilizes a Scikit-Learn `ColumnTransformer` embedded within a `Pipeline`.
- **Numerical Features**: Missing values are imputed using the median, and values are scaled using `StandardScaler`.
- **Categorical Features**: Missing values are imputed using the mode, and categorical text is encoded into numeric arrays using `OneHotEncoder`.
- The fitted preprocessor is saved as `preprocessor.pkl`. This exact object is required during inference to guarantee that live user input is scaled and encoded exactly like the training data.

### 4. Model Training & Selection (`model_trainer.py`)
Instead of assuming a single algorithm works best, this component trains an ensemble of powerful classifiers:
- **Random Forest**
- **XGBoost**
- **Logistic Regression**

**How it works**: It pulls hyperparameters dynamically from `config/params.yaml`. After training all models, it evaluates them against the test set. The model achieving the highest accuracy score is automatically declared the "Best Model" and is passed to the next stage.

### 5. Model Evaluation & ONNX Conversion (`model_evaluation.py`)
This is the most critical MLOps step in the pipeline.
- **Experiment Tracking**: It initializes a connection to **DagsHub** and starts an **MLflow** run. It logs the accuracy, precision, recall, and F1 score, alongside the exact parameters used by the winning model. This provides a historical audit trail of all experiments.
- **ONNX Conversion**: The winning model (e.g., Logistic Regression or Random Forest) is converted from a Python-specific object into an **ONNX (Open Neural Network Exchange)** graph. 
- **Why ONNX?** ONNX decouples the model from its training framework (scikit-learn/xgboost). Using `onnxruntime` during prediction offers drastically faster inference times, lower memory footprint, and the ability to deploy the model in non-Python environments (like C++ or Rust) in the future.

### 6. Serving Layer (`app.py` & `prediction_pipeline.py`)
The model is served using **FastAPI**, a modern, high-performance web framework.
- The `PredictPipeline` loads the ONNX model and the `preprocessor.pkl`. 
- User input is captured via an HTML form rendered by Jinja2, mapped to a Pandas DataFrame by the `CustomData` class, transformed, and fed into the ONNX Inference Session to return a real-time churn prediction.

---

## 🚀 Setup & Installation Guide

### Option 1: Local Virtual Environment Setup
Ideal for development and debugging.

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd telco_customer_churn
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -e .
   ```

### Option 2: Docker Containerization
Ideal for production deployment (AWS, GCP, Azure). 

The included `Dockerfile` ensures the application runs identically regardless of the host machine, bundling the OS, dependencies, and code into a single image.
1. **Build the Docker Image:**
   ```bash
   docker build -t churn-app:latest .
   ```
2. **Run the Container:**
   ```bash
   docker run -d -p 8000:8000 churn-app:latest
   ```
   The app will be accessible at `http://localhost:8000`.

---

## 🧪 Automated Testing

Testing ensures pipeline stability as the codebase evolves. Tests are located in the `tests/` directory and use the `pytest` framework.

- **`test_api.py`**: Verifies that the FastAPI endpoints (`/` and `/predict`) return `200 OK` responses and handle invalid data (e.g., 422 Unprocessable Entity) correctly using `TestClient`.
- **`test_data_models.py`**: Ensures the `CustomData` class correctly maps HTML form inputs into the exact Pandas DataFrame format expected by the `ColumnTransformer`.

**To run the tests locally:**
```bash
pytest -v tests/
```

---

## 🔄 CI/CD Workflow (GitHub Actions)

Continuous Integration and Continuous Deployment are configured via `.github/workflows/main.yaml`.

Every time code is pushed to the `main` branch, a GitHub Action runner is triggered. It performs the following sequential steps:
1. Provisions an Ubuntu environment and sets up Python 3.12.
2. Caches `pip` dependencies to dramatically reduce build times.
3. Installs project dependencies.
4. **Runs Pytest**: Executes the automated test suite. If any test fails, the pipeline halts, preventing broken code from reaching production.
5. **Executes the Training Pipeline**: If tests pass, it runs `python -m customer_churn.pipline.training_pipeline`, effectively training the model in the cloud and pushing the metrics to DagsHub.

---

## 🔐 Environment Variables

Because the GitHub Actions CI/CD runner automatically logs runs to DagsHub, it requires authentication.
You **must** configure the following secret in your GitHub Repository settings (`Settings > Secrets and variables > Actions`):
- `DAGSHUB_USER_TOKEN`: Your personal DagsHub API token.
