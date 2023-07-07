import copy
import os
import numpy as np
import pandas as pd
from airflow.models import Variable
from snowflake.connector.pandas_tools import pd_writer
from snowflake.sqlalchemy import URL
import snowflake.connector
from sqlalchemy import create_engine
import boto3
import logging

config = Variable.get("collections_model_dags_v1", deserialize_json=True)
s3 = boto3.resource("s3")
s3_bucket = config["s3_bucket"]

config_var = Variable.get("collections_model_dags_v1", deserialize_json=True)[
    "model_assets"
]

missing_value_num = config_var["missing_value_num"]
missing_value_cat = config_var["missing_value_cat"]
IV_threshold = config_var["IV_threshold"]  ### threshold for IV (IV should be accepted
var_threshold = config_var["var_threshold"]
ID_cols = config_var["ID_cols"]
input_path = config_var["input_path"]
data_path = config_var["data_path"]
model_path = config_var["model_path"]


def read_file(bucket_name, file_name):
    obj = s3.meta.client.get_object(Bucket=bucket_name, Key=file_name)
    return obj["Body"]

def get_raw_data(start_date, end_date,cur,query):
    sql_cmd = query.format(
        sd=start_date, ed=end_date
    )
    cur.execute(sql_cmd)
    df = pd.DataFrame(cur.fetchall())
    colnames = [desc[0] for desc in cur.description]
    df.columns = [i for i in colnames]
    return df

def truncate_table(identifier, dataset_name,cur):
    sql_cmd = f"TRUNCATE TABLE IF EXISTS analytics.kb_analytics.{identifier}_{dataset_name}_collectionsmodel_v1"
    try:
        cur.execute(sql_cmd)
    except Exception as error:
        logging.info(f"Error on truncate_table:{error}")
    return


def get_connector():
    conn = snowflake.connector.connect(
        user=config["user"],
        password=config["password"],
        account=config["account"],
        # user=os.environ.get('SNOWFLAKE_UNAME'),
        # password=os.environ.get('SNOWFLAKE_PASS'),
        # account=os.environ.get('SNOWFLAKE_ACCOUNT'),
        role=config["role"],
        warehouse=config["warehouse"],
        database=config["database"],
        insecure_mode=True,
    )
    return conn

def write_to_snowflake(data, identifier, dataset_name):
    data1 = data.copy()
    from sqlalchemy.types import (
        Boolean,
        Date,
        DateTime,
        Float,
        Integer,
        Interval,
        Text,
        Time,
    )

    dtype_dict = data1.dtypes.apply(lambda x: x.name).to_dict()
    for i in dtype_dict:
        if dtype_dict[i] == "datetime64[ns]":
            dtype_dict[i] = DateTime
        if dtype_dict[i] == "object":
            dtype_dict[i] = Text
        if dtype_dict[i] == "category":
            dtype_dict[i] = Text
        if dtype_dict[i] == "float64":
            dtype_dict[i] = Float
        if dtype_dict[i] == "float32":
            dtype_dict[i] = Float
        if dtype_dict[i] == "int64":
            dtype_dict[i] = Integer
        if dtype_dict[i] == "bool":
            dtype_dict[i] = Boolean
    dtype_dict
    engine = create_engine(
        URL(
            account=config["account"],
            user=config["user"],
            password=config["password"],
            # user=os.environ.get("SNOWFLAKE_UNAME"),
            # password=os.environ.get("SNOWFLAKE_PASS"),
            # account=os.environ.get("SNOWFLAKE_ACCOUNT"),
            database=config["database"],
            schema=config["schema"],
            warehouse=config["warehouse"],
            role=config["role"],
        )
    )

    # con = engine.raw_connection()
    data1.columns = map(lambda x: str(x).upper(), data1.columns)
    name = f"{identifier}_{dataset_name.lower()}_collectionsmodel_v1"
    data1.to_sql(
        name=name,
        con=engine,
        if_exists="replace",
        index=False,
        index_label=None,
        dtype=dtype_dict,
        method=pd_writer,
    )
    return