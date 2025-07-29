from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime
from lazypredict.Supervised import LazyClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

def load_classification_data():
    input_path = "/opt/airflow/dags/data/zillow_dataset.csv"
    staging_path = Variable.get("staging_area")

    df = pd.read_csv(input_path)
    df.to_parquet(f"{staging_path}/zillow_dataset.parquet")

def preprocess_classification_data():
    staging_path = Variable.get("staging_area")
    df = pd.read_parquet(f"{staging_path}/zillow_dataset.parquet")

    # Exemplo de coluna alvo (modifique conforme necessário)
    target_column = "status"  # Substitua por uma coluna categórica real do seu dataset

    # Remove linhas com NaN na target ou nas features
    df = df.dropna(subset=[target_column])
    df = df.select_dtypes(include=[np.number]).dropna()

    # Simulação de target categórica se não existir
    if target_column not in df.columns:
        df[target_column] = np.random.choice(["A", "B"], size=len(df))

    X = df.drop(columns=[target_column], errors='ignore')
    y = df[target_column]

    # Encoding da target se for string
    if y.dtype == "object" or y.dtype.name == "category":
        y = LabelEncoder().fit_transform(y)

    # Escala os dados
    X_scaled = StandardScaler().fit_transform(X)

    # Split para treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    np.save(f"{staging_path}/X_train_class.npy", X_train)
    np.save(f"{staging_path}/X_test_class.npy", X_test)
    np.save(f"{staging_path}/y_train_class.npy", y_train)
    np.save(f"{staging_path}/y_test_class.npy", y_test)

def run_lazy_classifier():
    staging_path = Variable.get("staging_area")

    X_train = np.load(f"{staging_path}/X_train_class.npy")
    X_test = np.load(f"{staging_path}/X_test_class.npy")
    y_train = np.load(f"{staging_path}/y_train_class.npy")
    y_test = np.load(f"{staging_path}/y_test_class.npy")

    clf = LazyClassifier(verbose=1, ignore_warnings=True)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    models.to_csv(f"{staging_path}/lazy_classification_results.csv")

default_args = {
    "start_date": datetime(2023, 1, 1),
}

with DAG(
    "classification_pipeline",
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
    tags=["ml", "classification"],
) as dag:

    task_load = PythonOperator(
        task_id="load_classification_data",
        python_callable=load_classification_data,
    )

    task_preprocess = PythonOperator(
        task_id="preprocess_classification_data",
        python_callable=preprocess_classification_data,
    )

    task_lazy_class = PythonOperator(
        task_id="run_lazy_classifier",
        python_callable=run_lazy_classifier,
    )

    task_load >> task_preprocess >> task_lazy_class
