#### initial declarations
import pandas as pd
import boto3
# airflow 
from airflow.models import Variable
from collections_model_utility_functions_v1 import read_file
import collections_model_utility_functions_v1 as uf


config = Variable.get("collections_model_dags_v1", deserialize_json=True)
s3 = boto3.resource("s3")
s3_bucket = config["s3_bucket"]

config_var = Variable.get("collections_model_dags_v1", deserialize_json=True)[
    "model_assets"
]


missing_value_num= -99999  ### missing value assignment
missing_value_cat="missing"
start_date="2023-04-01" ### Start date of modelling sample data
end_date="2023-05-25"   ### End date of modelling sample data
partition_date="2023-05-15" ## Train and OOT partition date
Val_date="2023-05-31" ### Validation data to be checked for testing model performance on unseen data
IV_threshold=0.01  ### threshold for IV (IV should be accepted
var_threshold=0.70  ### 75% of variantion in the features gets captured with PCA components

ID_cols=['USER_ID','LOAN_ID','FULL_DATE','AUTO_CURE_FLAG']

feature_list = pd.read_csv(
    read_file(uf.s3_bucket, uf.model_path + "cm_ac_model_v3_features.csv")
)