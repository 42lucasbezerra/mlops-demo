#!/usr/bin/env bash
set -e

# On AWS EC2 (Ubuntu 22.04)
# Install Python & venv
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv
python3 -m venv ~/mlflow-env
source ~/mlflow-env/bin/activate

# Install MLflow
pip install mlflow boto3

# Prepare directories
mkdir -p ~/mlflow/{mlruns,db}

# Set up environment vars
export MLFLOW_BACKEND_STORE_URI="sqlite:///~/mlflow/db/mlflow.db"
export MLFLOW_ARTIFACT_ROOT="~/mlflow/mlruns"

# Run server
mlflow server \
  --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
  --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
  --host 0.0.0.0 --port 5000