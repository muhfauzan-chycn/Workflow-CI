# Workflow-CI – MLflow Training Pipeline

Repository ini berisi workflow CI untuk melakukan retraining model machine learning
menggunakan MLflow Project dan GitHub Actions.

## Struktur
- MLProject/: berisi MLflow Project (training pipeline)
- .github/workflows/: workflow CI
- Dataset: hasil preprocessing dari eksperimen sebelumnya

## Workflow
Setiap push ke branch `main` akan:
1. Menjalankan MLflow Project
2. Melatih model
3. Menyimpan artefak MLflow
4. Build Docker image
5. Push image ke Docker Hub

## Docker Image
Image tersedia di Docker Hub:
`muhfuuzan/telco-churn-mlflow`
