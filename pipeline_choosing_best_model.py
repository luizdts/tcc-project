import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago

args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'description': "Uma DAG para automatizar um projeto de DS utilizando Airflow."
}

dags_path = "/opt/airflow/dags"

staging_area = Variable.get("staging_area")
engineering_team = Variable.get("email_engineering_team")
science_team = Variable.get("email_science_team")

def _load_dataset_properties_to_staging():
    df_ = pd.read_csv(dags_path+"/data/datasets/zillow/properties_2016.csv",
                      nrows=10000)
    
    df_.to_parquet(
        staging_area+"/properties_2016.parquet",
        index = False
    )

def _load_dataset_train_to_staging():

    df_ = pd.read_csv(dags_path+"/data/datasets/zillow/train_2016_v2.csv",
                      nrows=10000)
    
    df_.to_parquet(
        staging_area+"/train_2016_v2.parquet",
        index= False
    )

def _join_datasets():
    df_properties = pd.read_parquet(staging_area+"/properties_2016.parquet")
    df_train = pd.read_parquet(staging_area+"/train_2016_v2.parquet")

    df_final = df_properties.copy()
    df_final = df_final.merge(df_train, how='inner', on='parcelid')

    df_final.to_parquet(staging_area+"/zillow_dataset.parquet",
                        index=False)
    
def _check_remove_duplicated_rows():

    df_zillow = pd.read_parquet(staging_area+"/zillow_dataset.parquet")
    
    df_zillow.drop_duplicates(
        subset="parcelid",
        keep= "first",
        inplace= True
    )

    df_zillow.to_parquet(staging_area +"/zillow_dataset.parquet",
                         index=False)

def _drop_columns_percent_missing_values(**kwargs):
    df_zillow = pd.read_parquet(staging_area+"/zillow_dataset.parquet")

    percent_limit = kwargs['percent_limit']

    missing_var = [var for var in df_zillow.columns if df_zillow[var].isnull()
                   .sum() > 0]
    
    limit = np.abs((df_zillow.shape[0] * percent_limit))

    columns_drop = [var for var in missing_var if df_zillow[var].isnull()
                    .sum() > limit]
    
    df_zillow.drop(columns=columns_drop, axis= 1, inplace=True)

    df_zillow.to_parquet(
        staging_area+"/zillow_dataset.parquet",
        index=False
    )

def _transform_rescale_features():

    df_zillow = pd.read_parquet(staging_area+"/zillow_dataset.parquet")

    df_zillow['yeardifference'] = df_zillow['assessmentyear'] - df_zillow['yearbuilt']

    df_zillow[['latitude', 'longitude']] = (df_zillow[['latitude', 'longitude']])/(10**6)
    df_zillow['censustractandblock'] = (df_zillow['censustractandblock'])/(10**12)
    df_zillow['rawcensustractandblock'] = (df_zillow['rawcensustractandblock'])/(10**6)

    df_zillow.drop(columns=['assessmentyear', 'yearbuilt', 'transactiondate'], axis=1, inplace=True)

    df_zillow.to_parquet(staging_area+"/zillow_dataset.parquet",
                         index=False)
    
def _fill_missing_values():

    df_zillow = pd.read_parquet(staging_area+"/zillow_dataset.parquet")

    missing_var = [var for var in df_zillow.columns if df_zillow[var].isnull().sum() > 0]

    for var in missing_var:
        df_zillow[var] = df_zillow[var].fillna(df_zillow[var].mode()[0])

    df_zillow.to_parquet(
        staging_area+"/zillow_dataset.parquet",
        index=False
    )

def _encode_categorical_variables():

    df_zillow = pd.read_parquet(staging_area+"/zillow_dataset.parquet")

    categorical_variables = [var for var in df_zillow.columns if df_zillow[var].dtypes=='O']

    for i in range(len(categorical_variables)):
        var = categorical_variables[i]
        encoder = LabelEncoder()

        var_labels = encoder.fit_transform(df_zillow[var])
        var_mappings = {index: label for index, label in enumerate(encoder.classes_)}

        df_zillow[(var + '_labels')] = var_labels

        df_zillow.drop(columns=var, axis=1, inplace=True)
    
    df_zillow.to_parquet(
        staging_area+"/zillow_dataset.parquet",
        index=False
    )

def _drop_repetitive_useless_data():
    
    df_zillow = pd.read_parquet(staging_area+"/zillow_dataset.parquet")

    df_zillow.drop(
        columns=['censustractandblock',
                 'propertycountylandusecode_labels',
                 'parcelid'],
                 axis=1,
                 inplace=True
    )

    df_zillow.to_parquet(staging_area+"/zillow_dataset.parquet",
                         index= False)
    
def _preprocessing_separate_train_test():

    df_zillow = pd.read_parquet(staging_area+"/zillow_dataset.parquet")

    X = df_zillow.drop('logerror', axis=1)
    y = df_zillow['logerror']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)

    train_vars = [var for var in X_train.columns]

    scaler = StandardScaler()
    scaler.fit(X_train[train_vars])

    X_train[train_vars] = scaler.transform(X_train[train_vars])
    X_test[train_vars] = scaler.transform(X_test[train_vars])

    np.savetxt(staging_area+"/zillow_xtrain.csv", X_train, delimiter=",")
    
    np.savetxt(staging_area+"/zillow_ytrain.csv", y_train, delimiter=",")
    
    np.savetxt(staging_area+"/zillow_xtest.csv", X_test, delimiter=",")
    
    np.savetxt(staging_area+"/zillow_ytest.csv", y_test, delimiter=",")

def load_files_train_test_from_staging():
    X_train = np.loadtxt(staging_area+"/zillow_xtrain.csv", delimiter=",")
    y_train = np.loadtxt(staging_area+"/zillow_ytrain.csv", delimiter=",")
    X_test = np.loadtxt(staging_area+"/zillow_xtest.csv", delimiter=",")
    y_test = np.loadtxt(staging_area+"/zillow_ytest.csv", delimiter=",")
    
    return X_train, y_train, X_test, y_test

def train_model(estimator, X_train, y_train, X_test, y_test):

    estimator.fit(X_train, y_train)

    estimator_pred = estimator.predict(X_test)

    mean_abs_error = mean_absolute_error(y_test, estimator_pred)

    print('Mean Absolute Error: {}'.format(mean_abs_error))

    return mean_abs_error

def _train_model_regression_linear(ti):

    X_train, y_train, X_test, y_test = load_files_train_test_from_staging()

    linear_reg = LinearRegression()

    mean_abs_error = train_model(linear_reg, X_train, y_train, X_test, y_test)

    ti.xcom_push(key='mean_abs_error', value=mean_abs_error)

def _train_model_ada_boost_regressor(ti):

    X_train, y_train, X_test, y_test = load_files_train_test_from_staging()

    adaboost_reg = AdaBoostRegressor()

    mean_abs_error = train_model(adaboost_reg, X_train, y_train, X_test, y_test)

    ti.xcom_push(key='mean_abs_error', value=mean_abs_error)

def _train_model_gradient_boosting_regression(ti):

    X_train, y_train, X_test, y_test = load_files_train_test_from_staging()

    gb_reg = GradientBoostingRegressor()

    mean_abs_error = train_model(gb_reg, X_train, y_train, X_test, y_test)

    ti.xcom_push(key='mean_abs_error', value=mean_abs_error)

def _train_model_decision_tree_regressor(ti):

    X_train, y_train, X_test, y_test = load_files_train_test_from_staging()

    tree_reg = DecisionTreeRegressor()

    mean_abs_error = train_model(tree_reg, X_train, y_train, X_test, y_test)

    ti.xcom_push(key='mean_abs_error', value=mean_abs_error)

def _train_model_random_forest_regressor(ti):

    X_train, y_train, X_test, y_test = load_files_train_test_from_staging()

    forest_reg = RandomForestRegressor()

    mean_abs_error = train_model(forest_reg, X_train, y_train, X_test, y_test)

    ti.xcom_push(key='mean_abs_error', value=mean_abs_error)

def _choose_best_model(ti):
    models = [
        "LinearRegression",
        "AdaBoostRegressor",
        "GradientBoostingRegressor",
        "DecisionTreeRegressor",
        "RandomForestRegressor"
    ]

    metricas = ti.xcom_pull(
        key='mean_abs_error',
        task_ids=[
            "train_model_regression_linear",
            "train_model_ada_boost_regressor",
            "train_model_gradient_boosting_regression",
            "train_model_decision_tree_regressor",
            "train_model_random_forest_regressor"
        ]
    )
    index_best_model = metricas.index(min(metricas))

    print("Melhor modelo: {}, Score: {}".format(models[index_best_model],metricas[index_best_model]))

    ti.xcom_push(key='best_model', value=models[index_best_model])

def _final_model_train_dump(ti):

    X_train, y_train, X_test, y_test = load_files_train_test_from_staging()

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    best_model = ti.xcom_pull(
        key='best_model',
        task_ids=["choose_best_model"]
    )

    if best_model == "LinearRegression":
        estimator = LinearRegression()
    elif best_model == "AdaBoostRegressor":
        estimator = AdaBoostRegressor()
    elif best_model == "GradientBoostingRegressor":
        estimator == GradientBoostingRegressor()
    elif best_model == "DecisionTreeRegressor":
        estimator == DecisionTreeRegressor()
    else:
        estimator = RandomForestRegressor()

    print(estimator)
    estimator.fit(X, y)

    joblib.dump(estimator, staging_area+"/model.pkl")

with DAG('pipeline_projeto_housing_price', schedule_interval='@daily', default_args=args) as dag:

    load_dataset_properties_to_staging_task = PythonOperator(
        task_id = "load_dataset_properties_to_staging",
        python_callable= _load_dataset_properties_to_staging,
        email_on_failure = True,
        email = engineering_team
    )

    load_dataset_train_to_staging_task = PythonOperator(
        task_id = "load_dataset_train_to_staging",
        python_callable= _load_dataset_train_to_staging,
        email_on_failure = True,
        email = engineering_team
    )

    join_datasets_task = PythonOperator(
        task_id= "join_datasets",
        python_callable = _join_datasets,
        email_on_failure= True,
        email = engineering_team
    )

    check_remove_duplicated_rows_task = PythonOperator(
        task_id= "check_remove_duplcated_rows_task",
        python_callable = _check_remove_duplicated_rows,
        email_on_failure= True,
        email = engineering_team
    )

    drop_columns_percent_missing_task = PythonOperator(
        task_id = "drop_columns_percent_missing",
        python_callable= _drop_columns_percent_missing_values,
        op_kwargs={
            'percent_limit': 0.6
        },
        email_on_failure=True,
        email = engineering_team
    )

    transform_rescale_features_task = PythonOperator(
        task_id="transform_rescale_features",
        python_callable= _transform_rescale_features,
        email_on_failure= True,
        email = engineering_team
    )

    fill_missing_values_task = PythonOperator(
        task_id="fill_missing_values",
        python_callable= _fill_missing_values,
        email_on_failure= True,
        email = engineering_team
    )

    encode_categorical_variables_task = PythonOperator(
        task_id= "encode_categorical_variables",
        python_callable= _encode_categorical_variables,
        email_on_failure = True,
        email= engineering_team
    )

    drop_repetitive_useless_data_task = PythonOperator(
        task_id = "drop_repetitive_useless_data",
        python_callable = _drop_repetitive_useless_data,
        email_on_failure = True,
        email = engineering_team
    )

    preprocessing_separate_train_test_task = PythonOperator(
        task_id = "preprocessing_separate_train_test",
        python_callable= _preprocessing_separate_train_test,
        email_on_failure= True,
        email = science_team
    )

    train_model_regression_linear_task = PythonOperator(
        task_id = "train_model_regression_linear",
        python_callable = _train_model_regression_linear,
        email_on_failure = True,
        email = science_team
    )

    train_model_ada_boost_regressor_task = PythonOperator(
        task_id = "train_model_ada_boost_regressor",
        python_callable = _train_model_ada_boost_regressor,
        email_on_failure = True,
        email = science_team
    )

    train_model_gradient_boosting_regression_task = PythonOperator(
        task_id = "train_model_gradient_boosting_regression",
        python_callable = _train_model_gradient_boosting_regression,
        email_on_failure = True,
        email = science_team
    )

    train_model_decision_tree_regressor_task = PythonOperator(
        task_id = "train_model_decision_tree_regressor",
        python_callable = _train_model_decision_tree_regressor,
        email_on_failure = True,
        email = science_team
    )

    train_model_random_forest_regressor_task = PythonOperator(
        task_id = "train_model_random_forest_regressor",
        python_callable = _train_model_random_forest_regressor,
        email_on_failure = True,
        email = science_team
    )

    choose_best_model_task = PythonOperator(
        task_id = "choose_best_model",
        python_callable= _choose_best_model,
        email_on_failure=True,
        email= science_team
    )

    final_model_train_dump_task = PythonOperator(
        task_id = "final_model_train_dump",
        python_callable = _final_model_train_dump,
        email_on_failure = True,
        email = science_team
    )

[
    load_dataset_properties_to_staging_task,
    load_dataset_train_to_staging_task,
] >> join_datasets_task >> check_remove_duplicated_rows_task >> drop_columns_percent_missing_task >> transform_rescale_features_task >> fill_missing_values_task >> encode_categorical_variables_task >> drop_repetitive_useless_data_task >> preprocessing_separate_train_test_task >> [
    train_model_regression_linear_task,
    train_model_ada_boost_regressor_task,
    train_model_gradient_boosting_regression_task,
    train_model_decision_tree_regressor_task,
    train_model_random_forest_regressor_task
] >> choose_best_model_task >> final_model_train_dump_task