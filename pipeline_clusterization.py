from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime
from lazypredict.Supervised import LazyRegressor
from lazypredict.Unsupervised import LazyCluster
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

def load_clustering_data():
    input_path = "/opt/airflow/dags/data/zillow_dataset.csv"
    staging_path = Variable.get("staging_area")

    df = pd.read_csv(input_path)
    df.to_parquet(f"{staging_path}/zillow_dataset.parquet")

def preprocess_clustering_data():
    staging_path = Variable.get("staging_area")
    df = pd.read_parquet(f"{staging_path}/zillow_dataset.parquet")

    # Remove colunas não numéricas e NaNs
    df = df.select_dtypes(include=[np.number]).dropna()

    # Escalamento dos dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Salva os dados processados
    np.save(f"{staging_path}/X_cluster.npy", X_scaled)

def run_lazy_cluster():
    staging_path = Variable.get("staging_area")
    X = np.load(f"{staging_path}/X_cluster.npy")

    # Executa o LazyCluster
    cluster = LazyCluster(verbose=1, ignore_warnings=True)
    models = cluster.fit(X)

    # Salva os resultados
    results_df = models[0]
    results_df.to_csv(f"{staging_path}/lazy_cluster_results.csv")

default_args = {
    "start_date": datetime(2023, 1, 1),
}

with DAG(
    "clustering_pipeline",
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
    tags=["ml", "clustering"],
) as dag:

    task_load = PythonOperator(
        task_id="load_clustering_data",
        python_callable=load_clustering_data,
    )

    task_preprocess = PythonOperator(
        task_id="preprocess_clustering_data",
        python_callable=preprocess_clustering_data,
    )

    task_lazy_cluster = PythonOperator(
        task_id="run_lazy_cluster",
        python_callable=run_lazy_cluster,
    )

    task_load >> task_preprocess >> task_lazy_cluster
