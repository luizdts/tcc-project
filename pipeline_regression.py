import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago

args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'description': "Pipeline de regressão com LazyPredict testando 50 algoritmos"
}

dags_path = "/opt/airflow/dags"
staging_area = Variable.get("staging_area")

# 1. Carrega os dados do CSV
def _load_regression_data():
    df = pd.read_csv(f"{dags_path}/data/zillow_dataset.csv")
    df.to_parquet(f"{staging_area}/zillow_dataset.parquet", index=False)

# 2. Pré-processa e divide os dados
def _preprocess_and_split():
    df = pd.read_parquet(f"{staging_area}/zillow_dataset.parquet")
    df.dropna(inplace=True)
    y = df['logerror']
    X = df.drop(columns=['logerror'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    np.save(f"{staging_area}/x_train.npy", X_train_scaled)
    np.save(f"{staging_area}/x_test.npy", X_test_scaled)
    np.save(f"{staging_area}/y_train.npy", y_train.to_numpy())
    np.save(f"{staging_area}/y_test.npy", y_test.to_numpy())

# 3. Executa LazyRegressor testando diversos algoritmos
def _run_lazy_regressor():
    X_train = np.load(f"{staging_area}/x_train.npy")
    X_test = np.load(f"{staging_area}/x_test.npy")
    y_train = np.load(f"{staging_area}/y_train.npy")
    y_test = np.load(f"{staging_area}/y_test.npy")

    reg = LazyRegressor(
        verbose=1,
        ignore_warnings=True,
        custom_metric=None
    )

    # Isso testa 40–50 algoritmos automaticamente
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    # Salva os resultados em CSV
    models.to_csv(f"{staging_area}/regression_models_report.csv")
    predictions.to_csv(f"{staging_area}/regression_predictions_sample.csv")

with DAG('pipeline_regression_lazypredict', schedule_interval=None, default_args=args) as dag:

    load_data = PythonOperator(
        task_id="load_regression_data",
        python_callable=_load_regression_data
    )

    preprocess_data = PythonOperator(
        task_id="preprocess_and_split",
        python_callable=_preprocess_and_split
    )

    run_lazy = PythonOperator(
        task_id="run_lazy_regressor",
        python_callable=_run_lazy_regressor
    )

    load_data >> preprocess_data >> run_lazy
