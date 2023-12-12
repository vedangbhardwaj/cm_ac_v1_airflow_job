import pandas as pd
import numpy as np
import collections_model_get_data_v1 as gd
from collections_model_utility_functions_v1 import (
    get_connector,
    get_raw_data,
    truncate_table,
    write_to_snowflake,
    read_file,
)
import collections_model_utility_functions_v1 as uf
import copy 

# Model and performance evaluation
from sklearn.metrics import roc_curve, auc, precision_score, recall_score

## unpickling the model
import pickle
from datetime import datetime, timedelta

conn = get_connector()
cur = conn.cursor()


def trigger_cmd(sql_query, conn):
    data = pd.read_sql(sql_query, con=conn)
    if len(data) == 0:
        raise ValueError("Data shape not correct")


def identify_column_type(series):
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    elif pd.api.types.is_categorical_dtype(series):
        return "categorical"
    else:
        return "other"


def remove_inconsistent_values(df, feature_list):
    for column in feature_list:
        column_type = identify_column_type(df[column])
        if column_type == "numeric":
            df[column] = pd.to_numeric(df[column], errors="coerce")
            # df = df.dropna(subset=[column])
        elif column_type == "categorical":
            # df[column] = df[column].astype('category')
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            # Handle other column types if needed
            df[column] = pd.to_numeric(df[column], errors="coerce")
            # pass
    return df


# def split_data():
#     trigger_cmd(gd.split_dpd_today, conn=conn)
#     trigger_cmd(gd.merge_split_dpd_today, conn=conn)
#     return


# split_data()

'''
Things to be taken care of while changing the model:
 1. Change feature creation logic if any new features are added.
 1. Change csv file names and model pickle file 
 2. Create new merge table basis independent table writes 
 3. Update merge code basis the columns present in the table 
 4. Check if no-extra column is added due to any logic change in the data split or otherwise.
'''

query = """
    select  
        *
    from ANALYTICS.DBT_PROD.COLLECTIONS_MODEL_DS_V1
    --ANALYTICS.KB_ANALYTICS.AUTO_CURE_MODEL_FEATURE_BASE_CM_AC_TEST_V1_7
    -- ANALYTICS.DBT_PROD.COLLECTIONS_MODEL_DS_V1
    where date(FULL_DATE)>=date('{sd}')
    and date(FULL_DATE)<date('{ed}')
    """

# new logic - 0.12 fixed threshold, after 0.2 BAU, b/w 0.12 to 0.2 - 75% random call and 25% random not call

def model_run():
    # start_date = datetime.now().strftime("%Y-%m-%d")
    # end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = '2023-12-02'
    end_date = '2023-12-03'

    # get data
    data = get_raw_data(start_date, end_date, cur, query)

    print(f"*********** full data shape : {data.shape} ***********")

    change_list = ["LOAN_TYPE", "USER_TYPE", "VERIFICATION_TYPE", "PINCODE_RISK_FLAG"]

    def change_list_fn(data):
        for column in change_list:
            if gd.var_type(column) == "Categorical":
                # print(column)
                data[column] = data[column].fillna("MISSING").apply(lambda x: x.upper())
        return data

    data = change_list_fn(data)

    def encode_categ(data):
        category_loan_type_mapping = {"FRESH": 0, "RENEWAL": 1, "MISSING": 0}
        data["LOAN_TYPE_ENCODED"] = data["LOAN_TYPE"].map(category_loan_type_mapping)

        category_user_type_mapping = {"KB USER": 1, "NON-KB USER": 0, "MISSING": 0}
        data["USER_TYPE_ENCODED"] = data["USER_TYPE"].map(category_user_type_mapping)

        category_verification_type_mapping = {
            "PHYSICALLY VERIFIED": 2,
            "VIDEO VERIFIED": 1,
            "MISSING": 0,
        }
        data["VERIFICATION_TYPE_ENCODED"] = data["VERIFICATION_TYPE"].map(
            category_verification_type_mapping
        )

        category_riskpincode_type_mapping = {
            "HIGH": 2,
            "MEDIUM": 1,
            "LOW": 0,
            "MISSING": 0,
        }
        data["PINCODE_RISK_FLAG_ENCODED"] = data["PINCODE_RISK_FLAG"].map(
            category_riskpincode_type_mapping
        )

        return data

    data = encode_categ(data)

    # pick only selected features utilized in model
    feature_list = pd.read_csv(
        read_file(
            uf.s3_bucket, uf.model_path + "XGBoost_feature_list_100_prev_v3_t1_pos.csv"
        )
    )["variables"]
    # select only cols used in model
    data_filter = data[list(feature_list)]

    # remove_inconsistent_values(data)
    data_filter = remove_inconsistent_values(data_filter, feature_list)

    data_filter = gd.missing_ind_convert_num(data_filter)
    # unpickle cm_ac_vanilla_v1 model
    model = pickle.loads(
        uf.s3.Bucket(uf.s3_bucket)
        .Object(f"{uf.model_path}Vanilla_model_v3_4_filter_t1_pos.pkl")
        .get()["Body"]
        .read()
    )

    # run predictions and store the result
    data["Predicted_probability"] = model.predict_proba(data_filter)[:, 1]

    # finding threshold value for 35% prediction coverage
    # threshold_value = uf.find_threshold(data, uf.cut_off, "DS")
    threshold_value = 0.2

    # converting threshold cut to binary
    data["Predicted_binary_verdict"] = np.where(
        data["Predicted_probability"] >= threshold_value, 1, 0
    )
    # BAD FLAG
    # data['BAD_FLAG'] = np.where(data['NEXT_DAY_DPD']==2,1,0)

    # data['BAD_FLAG'].value_counts()

    # store only relevant info:
    data_write = data[
        list(feature_list)
        + [
            "LOAN_ID",
            "USER_ID",
            "FULL_DATE",
            "Predicted_probability",
            "Predicted_binary_verdict",
            "AUTO_CURE_FLAG",
        ]
    ]
    # precision = precision_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
    # Calculate recall
    # recall = recall_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
    # print("Precision Val data  {:.4f}%".format(precision*100))
    # print("Recall Val data {:.4f}%".format(recall*100))
    # fpr, tpr, thresholds = roc_curve(data_write['BAD_FLAG'], data_write["Predicted_probability"])
    # roc_auc = auc(fpr, tpr)
    # GINI_Val = (2 * roc_auc) - 1
    # print(f'Gini on Val data with XGBoost: {GINI_Val}')
    # write back to snowflake:

    # truncate_table("dod_results", "cm_ac_vanilla_v1".lower(), cur)
    # # write temp DoD table:
    # write_to_snowflake(data_write, "dod_results", "cm_ac_vanilla_v1".lower())

    get_split_data = """
        select 
            *
        from 
            ANALYTICS.KB_ANALYTICS.MASTER_DAILY_SPLIT_AB_TEST_COLLECTIONS_MODEL_DPD1
        where
            1=1
            and MODEL = 'DATA_SCIENCE'
            and UPDATE_DT = '2023-12-02'
    """

    split_data = get_raw_data(start_date, end_date, cur, get_split_data)
   
    # this line to be removed in stage repo
    # split_data['TOTAL_ROWS'] = None

    print(f"*********** split data for DS mode shape : {split_data.shape} ***********")

    exclude = [
        "RISK_FLAG",
        "ONE_DPD_TELE",
        "FINAL_RISK_FLAG",
        "RISK_RANK",
        "PERCENTILE",
        "ROW_NUM",
        "MODEL",
        # "TOTAL_ROWS",
    ]

    # excluding extra columns
    split_a_b = copy.copy(split_data.drop(columns=exclude))
    split_a_b.columns = split_a_b.columns.str.upper()
    
    data_write.rename(columns={"FULL_DATE": "UPDATE_DT"}, inplace=True)
    # converting all columns to UPPER CASE
    data_write.columns = data_write.columns.str.upper()

    print(f"*********** 100% split data shape : {split_a_b.shape} ***********")

    # BAU split
    data_a = copy.copy(data_write[data_write["PREDICTED_PROBABILITY"]>threshold_value])
    print(f"*********** BAU split data shape : {data_a.shape} ***********")
    print(f"*********** NO CALL Experiment restructured % split data overall : {(data_a.shape[0]/data_write.shape[0])*100}% ***********")

    # NO CALL Experiment restructured split
    data_b = copy.copy(data_write[~data_write["LOAN_ID"].isin(data_a["LOAN_ID"])])
    print(f"*********** NO CALL Experiment restructured split data shape : {data_b.shape} ***********")
    print(f"*********** NO CALL Experiment restructured % split data overall : {(data_b.shape[0]/data_write.shape[0])*100}% ***********")

    # converting data a threshold cut to binary
    data_a["PREDICTED_BINARY_VERDICT"] = np.where(
        data_a["PREDICTED_PROBABILITY"] >= threshold_value, 1, 0
    )

    # Define the risk probability bands
    risk_probability_bands = np.array([(0.12, 0.13), (0.13, 0.14), (0.14, 0.15), (0.15, 0.16), (0.16, 0.17),(0.17,0.18),(0.18,0.19),(0.19,0.2)])

    # Create a data frame for storing the selected records
    selected_records_df = pd.DataFrame()

    # Iterate through each remaining risk probability band
    for risk_probability_band in risk_probability_bands:
        # Select the records within the current risk probability band
        records = data_b[data_b['PREDICTED_PROBABILITY'].between(risk_probability_band[0], risk_probability_band[1])]

        # Randomly select 60% of the records
        selected_records = records.sample(frac=0.75)

        # Add the selected records to the data frame
        selected_records_df = selected_records_df.append(selected_records)

    # Return the data frame with selected records
    print(selected_records_df.shape)

    # converting data b randomly selected 60%-40% population to binary
    data_b["PREDICTED_BINARY_VERDICT"] = np.where(
        data_b["PREDICTED_PROBABILITY"] >= threshold_value, 1, 0
    )
    data_b["PREDICTED_BINARY_VERDICT"] = np.where(data_b['LOAN_ID'].isin(selected_records_df['LOAN_ID']), 1, 0)

    data_a["GROUP_NAME"] = "BAU threshold 0.2"
    data_b["GROUP_NAME"] = "NO_CALL_EXPERIMENT_RESTRUCTURED threshold 0.12"

    data_a_b = pd.concat([data_a, data_b], axis=0)
    data_a_b.reset_index(inplace=True)

    print(
        f"*********** 100% split data shape : {data_a_b.shape} ***********"
    )
    print(
        f"*********** 100% split data shape - value counts : {data_a_b['GROUP_NAME'].value_counts()} ***********"
    )

    print(
        f"*********** BAU data split : \n {data_a['PREDICTED_BINARY_VERDICT'].value_counts()} ***********"
    )
    print(
        f"*********** NO CALL Experiment restructured data split : \n {data_b['PREDICTED_BINARY_VERDICT'].value_counts()}***********"
    )
    print(
        f"*********** Risky - Non-Risky data split overall : \n {data_a_b['PREDICTED_BINARY_VERDICT'].value_counts()}***********"
    )
    print(
        f"*********** Risky - Non-Risky data split overall percentage : \n {data_a_b['PREDICTED_BINARY_VERDICT'].value_counts(normalize=True)*100}***********"
    )

    # converting all columns to UPPER CASE
    data_a_b.columns = data_a_b.columns.str.upper()

    split_data_pred = split_a_b.merge(
        data_a_b[
            [
                "LOAN_ID",
                "UPDATE_DT",
                "PREDICTED_PROBABILITY",
                "PREDICTED_BINARY_VERDICT",
                "GROUP_NAME"
            ]
        ],
        how="inner",
        on=["LOAN_ID", "UPDATE_DT"],
    )
    split_data_pred_all = copy.copy(split_data_pred)

    split_data_pred_all["RISK_FLAG"] = np.where(
        split_data_pred["PREDICTED_BINARY_VERDICT"] == 0, "Non-Risky", "Risky"
    )

    print("*********** POST DATA MERGE WITH TOTF SNAP NEW ***********\n")
    print(
        f"*********** 100% split data shape - value counts : {split_data_pred_all.shape} ***********"
    )
    print(
        f"*********** 100% split data shape - value counts : {split_data_pred_all['GROUP_NAME'].value_counts()} ***********"
    )
    print(
        f"*********** PREDICTED_BINARY_VERDICT overall percentage : \n {split_data_pred_all['PREDICTED_BINARY_VERDICT'].value_counts(normalize=True)*100} ***********"
    )
    print(
        f"*********** BAU data split - PREDICTED_BINARY_VERDICT - value counts: \n {split_data_pred_all[split_data_pred_all['GROUP_NAME']=='BAU threshold 0.2']['PREDICTED_BINARY_VERDICT'].value_counts()} ***********"
    )
    print(
        f"*********** BAU data split - PREDICTED_BINARY_VERDICT - overall percentage: \n {split_data_pred_all[split_data_pred_all['GROUP_NAME']=='BAU threshold 0.2']['PREDICTED_BINARY_VERDICT'].value_counts(normalize=True)*100} ***********"
    )
    print(
        f"*********** NO CALL Experiment restructured data split - PREDICTED_BINARY_VERDICT : \n {split_data_pred_all[split_data_pred_all['GROUP_NAME']=='NO_CALL_EXPERIMENT_RESTRUCTURED threshold 0.12']['PREDICTED_BINARY_VERDICT'].value_counts(normalize=True)*100}***********"
    )

    bins = np.arange(0.12,0.21,0.01)
    bins = np.append(0,bins)
    bins
    bin_labels =  ['Bin 1', 'Bin 2', 'Bin 3', 'Bin 4', 'Bin 5']
    
    # data sanity checks
    data_check = copy.copy(split_data_pred_all[split_data_pred_all['GROUP_NAME']=='NO_CALL_EXPERIMENT_RESTRUCTURED threshold 0.12'])
    data_check['PROBABILITY_BIN'] = pd.cut(data_check['PREDICTED_PROBABILITY'], bins=bins)

    result_total_count = data_check.groupby('PROBABILITY_BIN')['PREDICTED_BINARY_VERDICT'].count()
    result_value_counts = data_check.groupby('PROBABILITY_BIN')['PREDICTED_BINARY_VERDICT'].value_counts()
    result_value_counts_percentage = data_check.groupby('PROBABILITY_BIN')['PREDICTED_BINARY_VERDICT'].value_counts(normalize=True)
    print(result_total_count.head(100))
    print(result_value_counts.head(100))
    print(result_value_counts_percentage.head(20))


    split_data_pred_all_write = split_data_pred_all[
        [
            "UPDATE_DT",
            "LOAN_ID",
            "TODAYS_EDI",
            "PREDICTED_PROBABILITY",
            "RISK_FLAG",
            "GROUP_NAME",
        ]
    ]
    print(
        f"*********** predicted split data write for DS mode shape  : {split_data_pred_all_write.shape} ***********"
    )

    split_data_pred_all_write.rename(
        columns={"PREDICTED_PROBABILITY": "PROBABILITY_OF_RISKY"}, inplace=True
    )

    truncate_table(
        "dod_results_ab_test_full_write_monitoring", "cm_ac_vanilla_v1".lower(), cur
    )
    write_to_snowflake(
        split_data_pred_all_write,
        "dod_results_ab_test_full_write_monitoring",
        "cm_ac_vanilla_v1".lower(),
    )

    split_data_pred_risky = split_data_pred.loc[
        split_data_pred["PREDICTED_BINARY_VERDICT"] == 1, :
    ]
    split_data_pred_risky_share = split_data_pred_risky.drop(
        columns=["PREDICTED_BINARY_VERDICT", "PREDICTED_PROBABILITY"]
    )

    split_data_pred_risky_share["DB_UPDATED_AT"] = split_data_pred_risky_share[
        "DB_UPDATED_AT"
    ].astype("str")

    print(
        f"*********** predicted split data for DS mode shape  : {split_data_pred.shape} ***********"
    )
    print(
        f"*********** predicted split data for DS mode  : {split_data_pred.head()} ***********"
    )
    print(
        f"*********** Risky predicted split data for DS mode shape  : {split_data_pred_risky_share.shape} ***********"
    )

    split_data_pred_risky_share["RUN_TIMESTAMP"] = pd.to_datetime(datetime.now())
    split_data_pred_risky_share["RUN_TIMESTAMP"] = split_data_pred_risky_share[
        "RUN_TIMESTAMP"
    ].astype("str")
   
    # removing GROUP_NAME from RISKY segment shared to TC team
    split_data_pred_risky_share.drop(columns=['GROUP_NAME'],inplace=True)

    truncate_table("dod_results_ab_test", "cm_ac_vanilla_v1".lower(), cur)
    write_to_snowflake(
        split_data_pred_risky_share, "dod_results_ab_test", "cm_ac_vanilla_v1".lower()
    )

    # writing to common combination table
    trigger_cmd(gd.write_to_combination_table, conn)

    # writing to master table for monitoring
    trigger_cmd(gd.write_split_full_data, conn)

    # merge back to master predicted result table for 65% predicted risky of 50% total
    trigger_cmd(gd.merge_predicted_split_data, conn)

    # merging prediction to final table
    trigger_cmd(gd.merge_predicted_data, conn)

    cur.close()


# new re-structured No-calling experiment - currently deployed model

# def model_run():
#     start_date = datetime.now().strftime("%Y-%m-%d")
#     end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

#     # get data
#     data = get_raw_data(start_date, end_date, cur, gd.query)

#     print(f"*********** full data shape : {data.shape} ***********")

#     change_list = ["LOAN_TYPE", "USER_TYPE", "VERIFICATION_TYPE", "PINCODE_RISK_FLAG"]

#     def change_list_fn(data):
#         for column in change_list:
#             if gd.var_type(column) == "Categorical":
#                 # print(column)
#                 data[column] = data[column].fillna("MISSING").apply(lambda x: x.upper())
#         return data

#     data = change_list_fn(data)

#     def encode_categ(data):
#         category_loan_type_mapping = {"FRESH": 0, "RENEWAL": 1, "MISSING": 0}
#         data["LOAN_TYPE_ENCODED"] = data["LOAN_TYPE"].map(category_loan_type_mapping)

#         category_user_type_mapping = {"KB USER": 1, "NON-KB USER": 0, "MISSING": 0}
#         data["USER_TYPE_ENCODED"] = data["USER_TYPE"].map(category_user_type_mapping)

#         category_verification_type_mapping = {
#             "PHYSICALLY VERIFIED": 2,
#             "VIDEO VERIFIED": 1,
#             "MISSING": 0,
#         }
#         data["VERIFICATION_TYPE_ENCODED"] = data["VERIFICATION_TYPE"].map(
#             category_verification_type_mapping
#         )

#         category_riskpincode_type_mapping = {
#             "HIGH": 2,
#             "MEDIUM": 1,
#             "LOW": 0,
#             "MISSING": 0,
#         }
#         data["PINCODE_RISK_FLAG_ENCODED"] = data["PINCODE_RISK_FLAG"].map(
#             category_riskpincode_type_mapping
#         )

#         return data

#     data = encode_categ(data)

#     # pick only selected features utilized in model
#     feature_list = pd.read_csv(
#         read_file(
#             uf.s3_bucket, uf.model_path + "XGBoost_feature_list_100_prev_v3_t1_pos.csv"
#         )
#     )["variables"]
#     # select only cols used in model
#     data_filter = data[list(feature_list)]

#     # remove_inconsistent_values(data)
#     data_filter = remove_inconsistent_values(data_filter, feature_list)

#     data_filter = gd.missing_ind_convert_num(data_filter)
#     # unpickle cm_ac_vanilla_v1 model
#     model = pickle.loads(
#         uf.s3.Bucket(uf.s3_bucket)
#         .Object(f"{uf.model_path}Vanilla_model_v3_4_filter_t1_pos.pkl")
#         .get()["Body"]
#         .read()
#     )

#     # run predictions and store the result
#     data["Predicted_probability"] = model.predict_proba(data_filter)[:, 1]

#     # finding threshold value for 35% prediction coverage
#     # threshold_value = uf.find_threshold(data, uf.cut_off, "DS")
#     threshold_value = 0.5

#     # converting threshold cut to binary
#     data["Predicted_binary_verdict"] = np.where(
#         data["Predicted_probability"] >= threshold_value, 1, 0
#     )
#     # BAD FLAG
#     # data['BAD_FLAG'] = np.where(data['NEXT_DAY_DPD']==2,1,0)

#     # data['BAD_FLAG'].value_counts()

#     # store only relevant info:
#     data_write = data[
#         list(feature_list)
#         + [
#             "LOAN_ID",
#             "USER_ID",
#             "FULL_DATE",
#             "Predicted_probability",
#             "Predicted_binary_verdict",
#             "AUTO_CURE_FLAG",
#         ]
#     ]
#     # precision = precision_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
#     # Calculate recall
#     # recall = recall_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
#     # print("Precision Val data  {:.4f}%".format(precision*100))
#     # print("Recall Val data {:.4f}%".format(recall*100))
#     # fpr, tpr, thresholds = roc_curve(data_write['BAD_FLAG'], data_write["Predicted_probability"])
#     # roc_auc = auc(fpr, tpr)
#     # GINI_Val = (2 * roc_auc) - 1
#     # print(f'Gini on Val data with XGBoost: {GINI_Val}')
#     # write back to snowflake:

#     truncate_table("dod_results", "cm_ac_vanilla_v1".lower(), cur)
#     # write temp DoD table:
#     write_to_snowflake(data_write, "dod_results", "cm_ac_vanilla_v1".lower())

#     split_data = get_raw_data(start_date, end_date, cur, gd.get_split_data)
   
#     # this line to be removed in stage repo
#     # split_data['TOTAL_ROWS'] = None

#     print(f"*********** split data for DS mode shape : {split_data.shape} ***********")

#     exclude = [
#         "RISK_FLAG",
#         "ONE_DPD_TELE",
#         "FINAL_RISK_FLAG",
#         "RISK_RANK",
#         "PERCENTILE",
#         "ROW_NUM",
#         "MODEL",
#         "TOTAL_ROWS",
#     ]

#     # excluding extra columns
#     split_a_b = copy.copy(split_data.drop(columns=exclude))
#     split_a_b.columns = split_a_b.columns.str.upper()
    
#     data_write.rename(columns={"FULL_DATE": "UPDATE_DT"}, inplace=True)
#     # converting all columns to UPPER CASE
#     data_write.columns = data_write.columns.str.upper()

#     print(f"*********** 100% split data shape : {split_a_b.shape} ***********")

#     # BAU split
#     data_a = copy.copy(data_write[data_write["PREDICTED_PROBABILITY"]>threshold_value])
#     print(f"*********** BAU split data shape : {data_a.shape} ***********")
#     print(f"*********** NO CALL Experiment restructured % split data overall : {(data_a.shape[0]/data_write.shape[0])*100}% ***********")

#     # NO CALL Experiment restructured split
#     data_b = copy.copy(data_write[~data_write["LOAN_ID"].isin(data_a["LOAN_ID"])])
#     print(f"*********** NO CALL Experiment restructured split data shape : {data_b.shape} ***********")
#     print(f"*********** NO CALL Experiment restructured % split data overall : {(data_b.shape[0]/data_write.shape[0])*100}% ***********")

#     # converting data a threshold cut to binary
#     data_a["PREDICTED_BINARY_VERDICT"] = np.where(
#         data_a["PREDICTED_PROBABILITY"] >= threshold_value, 1, 0
#     )

#     # Define the risk probability bands
#     risk_probability_bands = np.array([(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)])

#     # Create a data frame for storing the selected records
#     selected_records_df = pd.DataFrame()

#     # Iterate through each remaining risk probability band
#     for risk_probability_band in risk_probability_bands:
#         # Select the records within the current risk probability band
#         records = data_b[data_b['PREDICTED_PROBABILITY'].between(risk_probability_band[0], risk_probability_band[1])]

#         # Randomly select 60% of the records
#         selected_records = records.sample(frac=0.6)

#         # Add the selected records to the data frame
#         selected_records_df = selected_records_df.append(selected_records)

#     # Return the data frame with selected records
#     print(selected_records_df.shape)

#     # converting data b randomly selected 60%-40% population to binary
#     data_b["PREDICTED_BINARY_VERDICT"] = np.where(data_b['LOAN_ID'].isin(selected_records_df['LOAN_ID']), 1, 0)

#     data_a["GROUP_NAME"] = "BAU"
#     data_b["GROUP_NAME"] = "NO_CALL_EXPERIMENT_RESTRUCTURED"

#     data_a_b = pd.concat([data_a, data_b], axis=0)
#     data_a_b.reset_index(inplace=True)

#     print(
#         f"*********** 100% split data shape : {data_a_b.shape} ***********"
#     )
#     print(
#         f"*********** 100% split data shape - value counts : {data_a_b['GROUP_NAME'].value_counts()} ***********"
#     )

#     print(
#         f"*********** BAU data split : \n {data_a['PREDICTED_BINARY_VERDICT'].value_counts()} ***********"
#     )
#     print(
#         f"*********** NO CALL Experiment restructured data split : \n {data_b['PREDICTED_BINARY_VERDICT'].value_counts()}***********"
#     )
#     print(
#         f"*********** Risky - Non-Risky data split overall : \n {data_a_b['PREDICTED_BINARY_VERDICT'].value_counts()}***********"
#     )
#     print(
#         f"*********** Risky - Non-Risky data split overall percentage : \n {data_a_b['PREDICTED_BINARY_VERDICT'].value_counts(normalize=True)*100}***********"
#     )

#     # converting all columns to UPPER CASE
#     data_a_b.columns = data_a_b.columns.str.upper()

#     split_data_pred = split_a_b.merge(
#         data_a_b[
#             [
#                 "LOAN_ID",
#                 "UPDATE_DT",
#                 "PREDICTED_PROBABILITY",
#                 "PREDICTED_BINARY_VERDICT",
#                 "GROUP_NAME"
#             ]
#         ],
#         how="inner",
#         on=["LOAN_ID", "UPDATE_DT"],
#     )
#     split_data_pred_all = copy.copy(split_data_pred)

#     split_data_pred_all["RISK_FLAG"] = np.where(
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 0, "Non-Risky", "Risky"
#     )

#     print("*********** POST DATA MERGE WITH TOTF SNAP NEW ***********\n")
#     print(
#         f"*********** 100% split data shape - value counts : {split_data_pred_all.shape} ***********"
#     )
#     print(
#         f"*********** 100% split data shape - value counts : {split_data_pred_all['GROUP_NAME'].value_counts()} ***********"
#     )
#     print(
#         f"*********** PREDICTED_BINARY_VERDICT overall percentage : \n {split_data_pred_all['PREDICTED_BINARY_VERDICT'].value_counts(normalize=True)*100} ***********"
#     )
#     print(
#         f"*********** BAU data split - PREDICTED_BINARY_VERDICT - value counts: \n {split_data_pred_all[split_data_pred_all['GROUP_NAME']=='BAU']['PREDICTED_BINARY_VERDICT'].value_counts()} ***********"
#     )
#     print(
#         f"*********** BAU data split - PREDICTED_BINARY_VERDICT - overall percentage: \n {split_data_pred_all[split_data_pred_all['GROUP_NAME']=='BAU']['PREDICTED_BINARY_VERDICT'].value_counts(normalize=True)*100} ***********"
#     )
#     print(
#         f"*********** NO CALL Experiment restructured data split - PREDICTED_BINARY_VERDICT : \n {split_data_pred_all[split_data_pred_all['GROUP_NAME']=='NO_CALL_EXPERIMENT_RESTRUCTURED']['PREDICTED_BINARY_VERDICT'].value_counts(normalize=True)*100}***********"
#     )

#     bins = np.arange(0,0.6,0.1)
#     bin_labels =  ['Bin 1', 'Bin 2', 'Bin 3', 'Bin 4', 'Bin 5']
    
#     # data sanity checks
#     data_check = copy.copy(split_data_pred_all[split_data_pred_all['GROUP_NAME']=='NO_CALL_EXPERIMENT_RESTRUCTURED'])
#     data_check['PROBABILITY_BIN'] = pd.cut(data_check['PREDICTED_PROBABILITY'], bins=bins)

#     result_total_count = data_check.groupby('PROBABILITY_BIN')['PREDICTED_BINARY_VERDICT'].count()
#     result_value_counts = data_check.groupby('PROBABILITY_BIN')['PREDICTED_BINARY_VERDICT'].value_counts()
#     result_value_counts_percentage = data_check.groupby('PROBABILITY_BIN')['PREDICTED_BINARY_VERDICT'].value_counts(normalize=True)
#     print(result_total_count.head(100))
#     print(result_value_counts.head(100))
#     print(result_value_counts_percentage.head(10))


#     split_data_pred_all_write = split_data_pred_all[
#         [
#             "UPDATE_DT",
#             "LOAN_ID",
#             "TODAYS_EDI",
#             "PREDICTED_PROBABILITY",
#             "RISK_FLAG",
#             "GROUP_NAME",
#         ]
#     ]
#     print(
#         f"*********** predicted split data write for DS mode shape  : {split_data_pred_all_write.shape} ***********"
#     )

#     split_data_pred_all_write.rename(
#         columns={"PREDICTED_PROBABILITY": "PROBABILITY_OF_RISKY"}, inplace=True
#     )

#     truncate_table(
#         "dod_results_ab_test_full_write_monitoring", "cm_ac_vanilla_v1".lower(), cur
#     )
#     write_to_snowflake(
#         split_data_pred_all_write,
#         "dod_results_ab_test_full_write_monitoring",
#         "cm_ac_vanilla_v1".lower(),
#     )

#     split_data_pred_risky = split_data_pred.loc[
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 1, :
#     ]
#     split_data_pred_risky_share = split_data_pred_risky.drop(
#         columns=["PREDICTED_BINARY_VERDICT", "PREDICTED_PROBABILITY"]
#     )

#     split_data_pred_risky_share["DB_UPDATED_AT"] = split_data_pred_risky_share[
#         "DB_UPDATED_AT"
#     ].astype("str")

#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred.shape} ***********"
#     )
#     print(
#         f"*********** predicted split data for DS mode  : {split_data_pred.head()} ***********"
#     )
#     print(
#         f"*********** Risky predicted split data for DS mode shape  : {split_data_pred_risky_share.shape} ***********"
#     )

#     split_data_pred_risky_share["RUN_TIMESTAMP"] = pd.to_datetime(datetime.now())
#     split_data_pred_risky_share["RUN_TIMESTAMP"] = split_data_pred_risky_share[
#         "RUN_TIMESTAMP"
#     ].astype("str")
   
#     # removing GROUP_NAME from RISKY segment shared to TC team
#     split_data_pred_risky_share.drop(columns=['GROUP_NAME'],inplace=True)

#     truncate_table("dod_results_ab_test", "cm_ac_vanilla_v1".lower(), cur)
#     write_to_snowflake(
#         split_data_pred_risky_share, "dod_results_ab_test", "cm_ac_vanilla_v1".lower()
#     )

#     # writing to common combination table
#     trigger_cmd(gd.write_to_combination_table, conn)

#     # writing to master table for monitoring
#     trigger_cmd(gd.write_split_full_data, conn)

#     # merge back to master predicted result table for 65% predicted risky of 50% total
#     trigger_cmd(gd.merge_predicted_split_data, conn)

#     # merging prediction to final table
#     trigger_cmd(gd.merge_predicted_data, conn)

#     cur.close()





# currently deployed code 

# import pandas as pd
# import numpy as np
# import collections_model_get_data_v1 as gd
# from collections_model_utility_functions_v1 import (
#     get_connector,
#     get_raw_data,
#     truncate_table,
#     write_to_snowflake,
#     read_file,
# )
# import collections_model_utility_functions_v1 as uf
# import copy

# # Model and performance evaluation
# from sklearn.metrics import roc_curve, auc, precision_score, recall_score

# ## unpickling the model
# import pickle
# from datetime import datetime, timedelta

# conn = get_connector()
# cur = conn.cursor()


# def trigger_cmd(sql_query, conn):
#     data = pd.read_sql(sql_query, con=conn)
#     if len(data) == 0:
#         raise ValueError("Data shape not correct")


# def identify_column_type(series):
#     if pd.api.types.is_numeric_dtype(series):
#         return "numeric"
#     elif pd.api.types.is_categorical_dtype(series):
#         return "categorical"
#     else:
#         return "other"


# def remove_inconsistent_values(df, feature_list):
#     for column in feature_list:
#         column_type = identify_column_type(df[column])
#         if column_type == "numeric":
#             df[column] = pd.to_numeric(df[column], errors="coerce")
#             # df = df.dropna(subset=[column])
#         elif column_type == "categorical":
#             # df[column] = df[column].astype('category')
#             df[column] = pd.to_numeric(df[column], errors="coerce")
#         else:
#             # Handle other column types if needed
#             df[column] = pd.to_numeric(df[column], errors="coerce")
#             # pass
#     return df


# def split_data():
#     trigger_cmd(gd.split_dpd_today, conn=conn)
#     trigger_cmd(gd.merge_split_dpd_today, conn=conn)
#     return

# def model_run():
#     start_date = datetime.now().strftime("%Y-%m-%d")
#     end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

#     # get data
#     data = get_raw_data(start_date, end_date, cur, gd.query)

#     print(f"*********** full data shape : {data.shape} ***********")

#     change_list = ["LOAN_TYPE", "USER_TYPE", "VERIFICATION_TYPE", "PINCODE_RISK_FLAG"]

#     def change_list_fn(data):
#         for column in change_list:
#             if gd.var_type(column) == "Categorical":
#                 # print(column)
#                 data[column] = data[column].fillna("MISSING").apply(lambda x: x.upper())
#         return data

#     data = change_list_fn(data)

#     def encode_categ(data):
#         category_loan_type_mapping = {"FRESH": 0, "RENEWAL": 1, "MISSING": 0}
#         data["LOAN_TYPE_ENCODED"] = data["LOAN_TYPE"].map(category_loan_type_mapping)

#         category_user_type_mapping = {"KB USER": 1, "NON-KB USER": 0, "MISSING": 0}
#         data["USER_TYPE_ENCODED"] = data["USER_TYPE"].map(category_user_type_mapping)

#         category_verification_type_mapping = {
#             "PHYSICALLY VERIFIED": 2,
#             "VIDEO VERIFIED": 1,
#             "MISSING": 0,
#         }
#         data["VERIFICATION_TYPE_ENCODED"] = data["VERIFICATION_TYPE"].map(
#             category_verification_type_mapping
#         )

#         category_riskpincode_type_mapping = {
#             "HIGH": 2,
#             "MEDIUM": 1,
#             "LOW": 0,
#             "MISSING": 0,
#         }
#         data["PINCODE_RISK_FLAG_ENCODED"] = data["PINCODE_RISK_FLAG"].map(
#             category_riskpincode_type_mapping
#         )

#         return data

#     data = encode_categ(data)

#     # pick only selected features utilized in model
#     feature_list = pd.read_csv(
#         read_file(
#             uf.s3_bucket, uf.model_path + "XGBoost_feature_list_100_prev_v3_t1_pos.csv"
#         )
#     )["variables"]
#     # select only cols used in model
#     data_filter = data[list(feature_list)]

#     # remove_inconsistent_values(data)
#     data_filter = remove_inconsistent_values(data_filter, feature_list)

#     data_filter = gd.missing_ind_convert_num(data_filter)
#     # unpickle cm_ac_vanilla_v1 model
#     model = pickle.loads(
#         uf.s3.Bucket(uf.s3_bucket)
#         .Object(f"{uf.model_path}Vanilla_model_v3_4_filter_t1_pos.pkl")
#         .get()["Body"]
#         .read()
#     )

#     # run predictions and store the result
#     data["Predicted_probability"] = model.predict_proba(data_filter)[:, 1]

#     # finding threshold value for 35% prediction coverage
#     threshold_value = uf.find_threshold(data, uf.cut_off, "DS")

#     # converting threshold cut to binary
#     data["Predicted_binary_verdict"] = np.where(
#         data["Predicted_probability"] >= threshold_value, 1, 0
#     )

#     # BAD FLAG
#     # data['BAD_FLAG'] = np.where(data['NEXT_DAY_DPD']==2,1,0)

#     # data['BAD_FLAG'].value_counts()

#     # store only relevant info:
#     data_write = data[
#         list(feature_list)
#         + [
#             "LOAN_ID",
#             "USER_ID",
#             "FULL_DATE",
#             "Predicted_probability",
#             "Predicted_binary_verdict",
#             "AUTO_CURE_FLAG",
#         ]
#     ]
#     # precision = precision_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
#     # Calculate recall
#     # recall = recall_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
#     # print("Precision Val data  {:.4f}%".format(precision*100))
#     # print("Recall Val data {:.4f}%".format(recall*100))
#     # fpr, tpr, thresholds = roc_curve(data_write['BAD_FLAG'], data_write["Predicted_probability"])
#     # roc_auc = auc(fpr, tpr)
#     # GINI_Val = (2 * roc_auc) - 1
#     # print(f'Gini on Val data with XGBoost: {GINI_Val}')
#     # write back to snowflake:

#     truncate_table("dod_results", "cm_ac_vanilla_v1".lower(), cur)
#     # write temp DoD table:
#     write_to_snowflake(data_write, "dod_results", "cm_ac_vanilla_v1".lower())

#     split_data = get_raw_data(start_date, end_date, cur, gd.get_split_data)
#     print(f"*********** split data for DS mode shape : {split_data.shape} ***********")

#     exclude = [
#         "RISK_FLAG",
#         "ONE_DPD_TELE",
#         "FINAL_RISK_FLAG",
#         "RISK_RANK",
#         "PERCENTILE",
#         "ROW_NUM",
#         "MODEL",
#         "TOTAL_ROWS",
#     ]

#     split_data_1 = split_data.drop(columns=exclude)
#     # filter 25% data for A/B test - 0.4 threshold
#     split_a = copy.copy(split_data_1.sample(frac=0.25, random_state=42))
#     print(f"*********** 25% split data shape : {split_a.shape} ***********")

#     # filter 75% data for top 65 percentile
#     split_b = copy.copy(split_data_1[~split_data_1["LOAN_ID"].isin(split_a["LOAN_ID"])])
#     print(f"*********** 75% split data shape : {split_b.shape} ***********")

#     data_write.rename(columns={"FULL_DATE": "UPDATE_DT"}, inplace=True)
#     # converting all columns to UPPER CASE
#     data_write.columns = data_write.columns.str.upper()

#     split_a["GROUP_NAME"] = "DS_25"
#     split_b["GROUP_NAME"] = "DS_75"

#     split_a_b = pd.concat([split_a, split_b], axis=0)
#     print(f"*********** 100% split data shape : {split_a_b.shape} ***********")
#     print(
#         f"*********** 100% split data shape : {split_a_b['GROUP_NAME'].value_counts()} ***********"
#     )

#     # converting Predicted_probability to original
#     data_write.rename(
#         columns={"PREDICTED_PROBABILITY": "Predicted_probability"}, inplace=True
#     )

#     # filter 25% data for A/B test - 0.4 threshold
#     data_a = copy.copy(data_write[data_write["LOAN_ID"].isin(split_a["LOAN_ID"])])
#     print(f"*********** 25% split data shape : {data_a.shape} ***********")

#     # filter 75% data for top 65 percentile
#     data_b = copy.copy(data_write[data_write["LOAN_ID"].isin(split_b["LOAN_ID"])])
#     print(f"*********** 75% split data shape : {data_b.shape} ***********")

#     # finding threshold value for 35% prediction coverage
#     threshold_value = uf.find_threshold(data_b, uf.cut_off, "DS")

#     # converting data a threshold cut to binary
#     data_a["PREDICTED_BINARY_VERDICT"] = np.where(
#         data_a["Predicted_probability"] >= 0.4, 1, 0
#     )

#     # converting data b threshold cut to binary
#     data_b["PREDICTED_BINARY_VERDICT"] = np.where(
#         data_b["Predicted_probability"] >= threshold_value, 1, 0
#     )

#     data_a_b = pd.concat([data_a, data_b], axis=0)
#     data_a_b.reset_index(inplace=True)
#     print(
#         f"*********** A data split : \n {data_a['PREDICTED_BINARY_VERDICT'].value_counts()} ***********"
#     )
#     print(
#         f"*********** B data split : \n {data_b['PREDICTED_BINARY_VERDICT'].value_counts()}***********"
#     )
#     print(
#         f"*********** A/B data split : \n {data_a_b['PREDICTED_BINARY_VERDICT'].value_counts()}***********"
#     )
#     print(f"*********** Post A/B split data merge : {data_a_b.shape} ***********")

#     # converting all columns to UPPER CASE
#     data_a_b.columns = data_a_b.columns.str.upper()

#     split_data_pred = split_a_b.merge(
#         data_a_b[
#             [
#                 "LOAN_ID",
#                 "UPDATE_DT",
#                 "PREDICTED_PROBABILITY",
#                 "PREDICTED_BINARY_VERDICT",
#             ]
#         ],
#         how="inner",
#         on=["LOAN_ID", "UPDATE_DT"],
#     )
#     split_data_pred_all = copy.copy(split_data_pred)
#     # split_data_pred_all["GROUP_NAME"] = "Treatment"

#     split_data_pred_all["RISK_FLAG"] = np.where(
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 0, "Non-Risky", "Risky"
#     )
#     split_data_pred_all_write = split_data_pred_all[
#         [
#             "UPDATE_DT",
#             "LOAN_ID",
#             "TODAYS_EDI",
#             "PREDICTED_PROBABILITY",
#             "RISK_FLAG",
#             "GROUP_NAME",
#         ]
#     ]
#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred_all_write.shape} ***********"
#     )

#     split_data_pred_all_write.rename(
#         columns={"PREDICTED_PROBABILITY": "PROBABILITY_OF_RISKY"}, inplace=True
#     )

#     truncate_table(
#         "dod_results_ab_test_full_write_monitoring", "cm_ac_vanilla_v1".lower(), cur
#     )
#     write_to_snowflake(
#         split_data_pred_all_write,
#         "dod_results_ab_test_full_write_monitoring",
#         "cm_ac_vanilla_v1".lower(),
#     )

#     split_data_pred_risky = split_data_pred.loc[
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 1, :
#     ]
#     split_data_pred_risky_share = split_data_pred_risky.drop(
#         columns=["PREDICTED_BINARY_VERDICT", "PREDICTED_PROBABILITY"]
#     )

#     split_data_pred_risky_share["DB_UPDATED_AT"] = split_data_pred_risky_share[
#         "DB_UPDATED_AT"
#     ].astype("str")

#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred.shape} ***********"
#     )
#     print(
#         f"*********** predicted split data for DS mode  : {split_data_pred.head()} ***********"
#     )
#     print(
#         f"*********** Risky predicted split data for DS mode shape  : {split_data_pred_risky_share.shape} ***********"
#     )

#     split_data_pred_risky_share["RUN_TIMESTAMP"] = pd.to_datetime(datetime.now())
#     split_data_pred_risky_share["RUN_TIMESTAMP"] = split_data_pred_risky_share[
#         "RUN_TIMESTAMP"
#     ].astype("str")
   
#     # removing GROUP_NAME from RISKY segment shared to TC team
#     split_data_pred_risky_share.drop(columns=['GROUP_NAME'],inplace=True)

#     truncate_table("dod_results_ab_test", "cm_ac_vanilla_v1".lower(), cur)
#     write_to_snowflake(
#         split_data_pred_risky_share, "dod_results_ab_test", "cm_ac_vanilla_v1".lower()
#     )

#     # writing to common combination table
#     trigger_cmd(gd.write_to_combination_table, conn)

#     # writing to master table for monitoring
#     trigger_cmd(gd.write_split_full_data, conn)

#     # merge back to master predicted result table for 65% predicted risky of 50% total
#     trigger_cmd(gd.merge_predicted_split_data, conn)

#     # merging prediction to final table
#     trigger_cmd(gd.merge_predicted_data, conn)

#     cur.close()

# current deployed logic 

# def model_run():
#     start_date = datetime.now().strftime("%Y-%m-%d")
#     end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

#     # get data
#     data = get_raw_data(start_date, end_date, cur, gd.query)

#     print(f"*********** full data shape : {data.shape} ***********")

#     change_list = ["LOAN_TYPE", "USER_TYPE", "VERIFICATION_TYPE", "PINCODE_RISK_FLAG"]

#     def change_list_fn(data):
#         for column in change_list:
#             if gd.var_type(column) == "Categorical":
#                 # print(column)
#                 data[column] = data[column].fillna("MISSING").apply(lambda x: x.upper())
#         return data

#     data = change_list_fn(data)

#     def encode_categ(data):
#         category_loan_type_mapping = {"FRESH": 0, "RENEWAL": 1, "MISSING": 0}
#         data["LOAN_TYPE_ENCODED"] = data["LOAN_TYPE"].map(category_loan_type_mapping)

#         category_user_type_mapping = {"KB USER": 1, "NON-KB USER": 0, "MISSING": 0}
#         data["USER_TYPE_ENCODED"] = data["USER_TYPE"].map(category_user_type_mapping)

#         category_verification_type_mapping = {
#             "PHYSICALLY VERIFIED": 2,
#             "VIDEO VERIFIED": 1,
#             "MISSING": 0,
#         }
#         data["VERIFICATION_TYPE_ENCODED"] = data["VERIFICATION_TYPE"].map(
#             category_verification_type_mapping
#         )

#         category_riskpincode_type_mapping = {
#             "HIGH": 2,
#             "MEDIUM": 1,
#             "LOW": 0,
#             "MISSING": 0,
#         }
#         data["PINCODE_RISK_FLAG_ENCODED"] = data["PINCODE_RISK_FLAG"].map(
#             category_riskpincode_type_mapping
#         )

#         return data

#     data = encode_categ(data)

#     # pick only selected features utilized in model
#     feature_list = pd.read_csv(
#         read_file(
#             uf.s3_bucket, uf.model_path + "XGBoost_feature_list_100_prev_v3_t1_pos.csv"
#         )
#     )["variables"]
#     # select only cols used in model
#     data_filter = data[list(feature_list)]

#     # remove_inconsistent_values(data)
#     data_filter = remove_inconsistent_values(data_filter, feature_list)

#     data_filter = gd.missing_ind_convert_num(data_filter)
#     # unpickle cm_ac_vanilla_v1 model
#     model = pickle.loads(
#         uf.s3.Bucket(uf.s3_bucket)
#         .Object(f"{uf.model_path}Vanilla_model_v3_4_filter_t1_pos.pkl")
#         .get()["Body"]
#         .read()
#     )

#     # run predictions and store the result
#     data["Predicted_probability"] = model.predict_proba(data_filter)[:, 1]

#     # filter 25% data for A/B test - 0.4 threshold 
#     data_a = copy.copy(data.sample(frac=0.25,random_state=42))

#     # filter 75% data for top 65 percentile
#     data_b = data[~data['LOAN_ID'].isin(data_a['LOAN_ID'])]

#     # finding threshold value for 35% prediction coverage
#     threshold_value = uf.find_threshold(data, uf.cut_off, "DS")

#     # converting threshold cut to binary
#     data["Predicted_binary_verdict"] = np.where(
#         data["Predicted_probability"] >= threshold_value, 1, 0
#     )

#     # BAD FLAG
#     # data['BAD_FLAG'] = np.where(data['NEXT_DAY_DPD']==2,1,0)

#     # data['BAD_FLAG'].value_counts()

#     # store only relevant info:
#     data_write = data[
#         list(feature_list)
#         + [
#             "LOAN_ID",
#             "USER_ID",
#             "FULL_DATE",
#             "Predicted_probability",
#             "Predicted_binary_verdict",
#             "AUTO_CURE_FLAG",
#         ]
#     ]
#     # precision = precision_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
#     # Calculate recall
#     # recall = recall_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
#     # print("Precision Val data  {:.4f}%".format(precision*100))
#     # print("Recall Val data {:.4f}%".format(recall*100))
#     # fpr, tpr, thresholds = roc_curve(data_write['BAD_FLAG'], data_write["Predicted_probability"])
#     # roc_auc = auc(fpr, tpr)
#     # GINI_Val = (2 * roc_auc) - 1
#     # print(f'Gini on Val data with XGBoost: {GINI_Val}')
#         # write back to snowflake:
    
#     truncate_table("dod_results", "cm_ac_vanilla_v1".lower(), cur)
#     # write temp DoD table:
#     write_to_snowflake(data_write, "dod_results", "cm_ac_vanilla_v1".lower())

#     split_data = get_raw_data(start_date, end_date, cur, gd.get_split_data)
#     print(f"*********** split data for DS mode shape : {split_data.shape} ***********")

#     exclude = [
#         "RISK_FLAG",
#         "ONE_DPD_TELE",
#         "FINAL_RISK_FLAG",
#         "RISK_RANK",
#         "PERCENTILE",
#         "ROW_NUM",
#         "MODEL",
#         "TOTAL_ROWS",
#     ]

#     split_data_1 = split_data.drop(columns=exclude)
#     data_write.rename(columns={"FULL_DATE": "UPDATE_DT"}, inplace=True)
#     # converting all columns to UPPER CASE
#     data_write.columns = data_write.columns.str.upper()

#     split_data_pred = split_data_1.merge(
#         data_write[
#             [
#                 "LOAN_ID",
#                 "UPDATE_DT",
#                 "PREDICTED_PROBABILITY",
#                 "PREDICTED_BINARY_VERDICT",
#             ]
#         ],
#         how="inner",
#         on=["LOAN_ID", "UPDATE_DT"],
#     )

#     split_data_pred_all = copy.copy(split_data_pred)
#     split_data_pred_all["GROUP_NAME"] = "Treatment"

#     split_data_pred_all["RISK_FLAG"] = np.where(
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 0, "Non-Risky", "Risky"
#     )
#     split_data_pred_all_write = split_data_pred_all[
#         [
#             "UPDATE_DT",
#             "LOAN_ID",
#             "TODAYS_EDI",
#             "PREDICTED_PROBABILITY",
#             "RISK_FLAG",
#             "GROUP_NAME",
#         ]
#     ]
#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred_all_write.shape} ***********"
#     )

#     split_data_pred_all_write.rename(
#         columns={"PREDICTED_PROBABILITY": "PROBABILITY_OF_RISKY"}, inplace=True
#     )

#     truncate_table(
#         "dod_results_ab_test_full_write_monitoring", "cm_ac_vanilla_v1".lower(), cur
#     )
#     write_to_snowflake(
#         split_data_pred_all_write,
#         "dod_results_ab_test_full_write_monitoring",
#         "cm_ac_vanilla_v1".lower(),
#     )

#     split_data_pred_risky = split_data_pred.loc[
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 1, :
#     ]
#     split_data_pred_risky_share = split_data_pred_risky.drop(
#         columns=["PREDICTED_BINARY_VERDICT", "PREDICTED_PROBABILITY"]
#     )

#     split_data_pred_risky_share["DB_UPDATED_AT"] = split_data_pred_risky_share[
#         "DB_UPDATED_AT"
#     ].astype("str")

#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred.shape} ***********"
#     )
#     print(
#         f"*********** predicted split data for DS mode  : {split_data_pred.head()} ***********"
#     )
#     print(
#         f"*********** Risky predicted split data for DS mode shape  : {split_data_pred_risky_share.shape} ***********"
#     )

#     split_data_pred_risky_share["RUN_TIMESTAMP"] = pd.to_datetime(datetime.now())
#     split_data_pred_risky_share["RUN_TIMESTAMP"] = split_data_pred_risky_share[
#         "RUN_TIMESTAMP"
#     ].astype("str")

#     truncate_table("dod_results_ab_test", "cm_ac_vanilla_v1".lower(), cur)
#     write_to_snowflake(
#         split_data_pred_risky_share, "dod_results_ab_test", "cm_ac_vanilla_v1".lower()
#     )

#     # writing to common combination table
#     trigger_cmd(gd.write_to_combination_table, conn)

#     # writing to master table for monitoring
#     trigger_cmd(gd.write_split_full_data, conn)

#     # merge back to master predicted result table for 65% predicted risky of 50% total
#     trigger_cmd(gd.merge_predicted_split_data, conn)

#     # merging prediction to final table
#     trigger_cmd(gd.merge_predicted_data, conn)

#     cur.close()


# def model_run():
#     start_date = datetime.now().strftime("%Y-%m-%d")
#     end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

#     # get data
#     data = get_raw_data(start_date, end_date, cur, gd.query)

#     print(f"*********** full data shape : {data.shape} ***********")

#     change_list = ["LOAN_TYPE", "USER_TYPE", "VERIFICATION_TYPE", "PINCODE_RISK_FLAG"]

#     def change_list_fn(data):
#         for column in change_list:
#             if gd.var_type(column) == "Categorical":
#                 # print(column)
#                 data[column] = data[column].fillna("MISSING").apply(lambda x: x.upper())
#         return data

#     data = change_list_fn(data)

#     def encode_categ(data):
#         category_loan_type_mapping = {"FRESH": 0, "RENEWAL": 1, "MISSING": 0}
#         data["LOAN_TYPE_ENCODED"] = data["LOAN_TYPE"].map(category_loan_type_mapping)

#         category_user_type_mapping = {"KB USER": 1, "NON-KB USER": 0, "MISSING": 0}
#         data["USER_TYPE_ENCODED"] = data["USER_TYPE"].map(category_user_type_mapping)

#         category_verification_type_mapping = {
#             "PHYSICALLY VERIFIED": 2,
#             "VIDEO VERIFIED": 1,
#             "MISSING": 0,
#         }
#         data["VERIFICATION_TYPE_ENCODED"] = data["VERIFICATION_TYPE"].map(
#             category_verification_type_mapping
#         )

#         category_riskpincode_type_mapping = {
#             "HIGH": 2,
#             "MEDIUM": 1,
#             "LOW": 0,
#             "MISSING": 0,
#         }
#         data["PINCODE_RISK_FLAG_ENCODED"] = data["PINCODE_RISK_FLAG"].map(
#             category_riskpincode_type_mapping
#         )

#         return data

#     data = encode_categ(data)

#     # pick only selected features utilized in model
#     feature_list = pd.read_csv(
#         read_file(
#             uf.s3_bucket, uf.model_path + "XGBoost_feature_list_100_prev_v3_t1_pos.csv"
#         )
#     )["variables"]
#     # select only cols used in model
#     data_filter = data[list(feature_list)]

#     # remove_inconsistent_values(data)
#     data_filter = remove_inconsistent_values(data_filter, feature_list)

#     data_filter = gd.missing_ind_convert_num(data_filter)
#     # unpickle cm_ac_vanilla_v1 model
#     model = pickle.loads(
#         uf.s3.Bucket(uf.s3_bucket)
#         .Object(f"{uf.model_path}Vanilla_model_v3_4_filter_t1_pos.pkl")
#         .get()["Body"]
#         .read()
#     )

#     # run predictions and store the result
#     data["Predicted_probability"] = model.predict_proba(data_filter)[:, 1]

#     # finding threshold value for 35% prediction coverage
#     threshold_value = uf.find_threshold(data, uf.cut_off, "DS")

#     # converting threshold cut to binary
#     data["Predicted_binary_verdict"] = np.where(
#         data["Predicted_probability"] >= threshold_value, 1, 0
#     )

#     # BAD FLAG
#     # data['BAD_FLAG'] = np.where(data['NEXT_DAY_DPD']==2,1,0)

#     # data['BAD_FLAG'].value_counts()

#     # store only relevant info:
#     data_write = data[
#         list(feature_list)
#         + [
#             "LOAN_ID",
#             "USER_ID",
#             "FULL_DATE",
#             "Predicted_probability",
#             "Predicted_binary_verdict",
#             "AUTO_CURE_FLAG",
#         ]
#     ]

#     # precision = precision_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
#     # Calculate recall
#     # recall = recall_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
#     # print("Precision Val data  {:.4f}%".format(precision*100))
#     # print("Recall Val data {:.4f}%".format(recall*100))

#     # fpr, tpr, thresholds = roc_curve(data_write['BAD_FLAG'], data_write["Predicted_probability"])
#     # roc_auc = auc(fpr, tpr)
#     # GINI_Val = (2 * roc_auc) - 1
#     # print(f'Gini on Val data with XGBoost: {GINI_Val}')

#     # write back to snowflake:
#     truncate_table("dod_results", "cm_ac_vanilla_v1".lower(), cur)
#     # write temp DoD table:
#     write_to_snowflake(data_write, "dod_results", "cm_ac_vanilla_v1".lower())

#     split_data = get_raw_data(start_date, end_date, cur, gd.get_split_data)
#     print(f"*********** split data for DS mode shape : {split_data.shape} ***********")

#     exclude = [
#         "RISK_FLAG",
#         "ONE_DPD_TELE",
#         "FINAL_RISK_FLAG",
#         "RISK_RANK",
#         "PERCENTILE",
#         "ROW_NUM",
#         "MODEL",
#         "TOTAL_ROWS",
#     ]

#     split_data_1 = split_data.drop(columns=exclude)
#     data_write.rename(columns={"FULL_DATE": "UPDATE_DT"}, inplace=True)
#     # converting all columns to UPPER CASE
#     data_write.columns = data_write.columns.str.upper()

#     split_data_pred = split_data_1.merge(
#         data_write[
#             [
#                 "LOAN_ID",
#                 "UPDATE_DT",
#                 "PREDICTED_PROBABILITY",
#                 "PREDICTED_BINARY_VERDICT",
#             ]
#         ],
#         how="inner",
#         on=["LOAN_ID", "UPDATE_DT"],
#     )

#     split_data_pred_all = copy.copy(split_data_pred)
#     split_data_pred_all["GROUP_NAME"] = "Treatment"

#     split_data_pred_all["RISK_FLAG"] = np.where(
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 0, "Non-Risky", "Risky"
#     )
#     split_data_pred_all_write = split_data_pred_all[
#         [
#             "UPDATE_DT",
#             "LOAN_ID",
#             "TODAYS_EDI",
#             "PREDICTED_PROBABILITY",
#             "RISK_FLAG",
#             "GROUP_NAME",
#         ]
#     ]

#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred_all_write.shape} ***********"
#     )

#     split_data_pred_all_write.rename(
#         columns={"PREDICTED_PROBABILITY": "PROBABILITY_OF_RISKY"}, inplace=True
#     )

#     truncate_table(
#         "dod_results_ab_test_full_write_monitoring", "cm_ac_vanilla_v1".lower(), cur
#     )
#     write_to_snowflake(
#         split_data_pred_all_write,
#         "dod_results_ab_test_full_write_monitoring",
#         "cm_ac_vanilla_v1".lower(),
#     )

#     split_data_pred_risky = split_data_pred.loc[
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 1, :
#     ]
#     split_data_pred_risky_share = split_data_pred_risky.drop(
#         columns=["PREDICTED_BINARY_VERDICT", "PREDICTED_PROBABILITY"]
#     )

#     split_data_pred_risky_share["DB_UPDATED_AT"] = split_data_pred_risky_share[
#         "DB_UPDATED_AT"
#     ].astype("str")

#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred.shape} ***********"
#     )
#     print(
#         f"*********** predicted split data for DS mode  : {split_data_pred.head()} ***********"
#     )
#     print(
#         f"*********** Risky predicted split data for DS mode shape  : {split_data_pred_risky_share.shape} ***********"
#     )

#     split_data_pred_risky_share["RUN_TIMESTAMP"] = pd.to_datetime(datetime.now())
#     split_data_pred_risky_share["RUN_TIMESTAMP"] = split_data_pred_risky_share[
#         "RUN_TIMESTAMP"
#     ].astype("str")

#     truncate_table("dod_results_ab_test", "cm_ac_vanilla_v1".lower(), cur)
#     write_to_snowflake(
#         split_data_pred_risky_share, "dod_results_ab_test", "cm_ac_vanilla_v1".lower()
#     )

#     # writing to common combination table
#     trigger_cmd(gd.write_to_combination_table, conn)

#     # writing to master table for monitoring
#     trigger_cmd(gd.write_split_full_data, conn)

#     # merge back to master predicted result table for 65% predicted risky of 50% total
#     trigger_cmd(gd.merge_predicted_split_data, conn)

#     # merging prediction to final table
#     trigger_cmd(gd.merge_predicted_data, conn)

#     cur.close()
#     conn.close()


# def model_run():
#     # start_date = datetime.now().strftime("%Y-%m-%d")
#     # end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
#     start_date = '2023-07-26'
#     end_date = '2023-07-27'

#     # get data
#     data = get_raw_data(start_date, end_date, cur, query)

#     print(f"*********** full data shape : {data.shape} ***********")

#     change_list = ["LOAN_TYPE", "USER_TYPE", "VERIFICATION_TYPE", "PINCODE_RISK_FLAG"]

#     def change_list_fn(data):
#         for column in change_list:
#             if gd.var_type(column) == "Categorical":
#                 # print(column)
#                 data[column] = data[column].fillna("MISSING").apply(lambda x: x.upper())
#         return data

#     data = change_list_fn(data)

#     def encode_categ(data):
#         category_loan_type_mapping = {"FRESH": 0, "RENEWAL": 1, "MISSING": 0}
#         data["LOAN_TYPE_ENCODED"] = data["LOAN_TYPE"].map(category_loan_type_mapping)

#         category_user_type_mapping = {"KB USER": 1, "NON-KB USER": 0, "MISSING": 0}
#         data["USER_TYPE_ENCODED"] = data["USER_TYPE"].map(category_user_type_mapping)

#         category_verification_type_mapping = {
#             "PHYSICALLY VERIFIED": 2,
#             "VIDEO VERIFIED": 1,
#             "MISSING": 0,
#         }
#         data["VERIFICATION_TYPE_ENCODED"] = data["VERIFICATION_TYPE"].map(
#             category_verification_type_mapping
#         )

#         category_riskpincode_type_mapping = {
#             "HIGH": 2,
#             "MEDIUM": 1,
#             "LOW": 0,
#             "MISSING": 0,
#         }
#         data["PINCODE_RISK_FLAG_ENCODED"] = data["PINCODE_RISK_FLAG"].map(
#             category_riskpincode_type_mapping
#         )

#         return data

#     data = encode_categ(data)

#     # pick only selected features utilized in model
#     feature_list = pd.read_csv(
#         read_file(
#             uf.s3_bucket, uf.model_path + "XGBoost_feature_list_100_prev_v3_t1_pos.csv"
#         )
#     )["variables"]
#     # select only cols used in model
#     data_filter = data[list(feature_list)]

#     # remove_inconsistent_values(data)
#     data_filter = remove_inconsistent_values(data_filter, feature_list)

#     data_filter = gd.missing_ind_convert_num(data_filter)
#     # unpickle cm_ac_vanilla_v1 model
#     model = pickle.loads(
#         uf.s3.Bucket(uf.s3_bucket)
#         .Object(f"{uf.model_path}Vanilla_model_v3_4_filter_t1_pos.pkl")
#         .get()["Body"]
#         .read()
#     )

#     # run predictions and store the result
#     data["Predicted_probability"] = model.predict_proba(data_filter)[:, 1]

#     # finding threshold value for 35% prediction coverage
#     threshold_value = uf.find_threshold(data, uf.cut_off, "DS")

#     # converting threshold cut to binary
#     data["Predicted_binary_verdict"] = np.where(
#         data["Predicted_probability"] >= threshold_value, 1, 0
#     )

#     # BAD FLAG
#     data['BAD_FLAG'] = np.where(data['NEXT_DAY_DPD']==2,1,0)

#     data['BAD_FLAG'].value_counts()

#     # store only relevant info:
#     data_write = data[
#         list(feature_list)
#         + [
#             "LOAN_ID",
#             "USER_ID",
#             "FULL_DATE",
#             "Predicted_probability",
#             "Predicted_binary_verdict",
#             "AUTO_CURE_FLAG"
#         ]
#     ]

#     precision = precision_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
#     # Calculate recall
#     recall = recall_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
#     print("Precision Val data  {:.4f}%".format(precision*100))
#     print("Recall Val data {:.4f}%".format(recall*100))

#     fpr, tpr, thresholds = roc_curve(data_write['BAD_FLAG'], data_write["Predicted_probability"])
#     roc_auc = auc(fpr, tpr)
#     GINI_Val = (2 * roc_auc) - 1
#     print(f'Gini on Val data with XGBoost: {GINI_Val}')

#     # write back to snowflake:
#     truncate_table("dod_results", "cm_ac_vanilla_v1_temp".lower(), cur)
#     # write temp DoD table:
#     write_to_snowflake(data_write, "dod_results", "cm_ac_vanilla_v1_temp".lower())

#     split_data = get_raw_data(start_date, end_date, cur, gd.get_split_data)
#     print(f"*********** split data for DS mode shape : {split_data.shape} ***********")

#     exclude = [
#         "RISK_FLAG",
#         "ONE_DPD_TELE",
#         "FINAL_RISK_FLAG",
#         "RISK_RANK",
#         "PERCENTILE",
#         "ROW_NUM",
#         "MODEL",
#         "TOTAL_ROWS",
#     ]

#     split_data_1 = split_data.drop(columns=exclude)
#     data_write.rename(columns={"FULL_DATE": "UPDATE_DT"}, inplace=True)
#     # converting all columns to UPPER CASE
#     data_write.columns = data_write.columns.str.upper()

#     split_data_pred = split_data_1.merge(
#         data_write[
#             [
#                 "LOAN_ID",
#                 "UPDATE_DT",
#                 "PREDICTED_PROBABILITY",
#                 "PREDICTED_BINARY_VERDICT",
#             ]
#         ],
#         how="inner",
#         on=["LOAN_ID", "UPDATE_DT"],
#     )

#     split_data_pred_all = copy.copy(split_data_pred)
#     split_data_pred_all["GROUP_NAME"] = "Treatment"

#     split_data_pred_all["RISK_FLAG"] = np.where(
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 0, "Non-Risky", "Risky"
#     )
#     split_data_pred_all_write = split_data_pred_all[
#         [
#             "UPDATE_DT",
#             "LOAN_ID",
#             "TODAYS_EDI",
#             "PREDICTED_PROBABILITY",
#             "RISK_FLAG",
#             "GROUP_NAME",
#         ]
#     ]

#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred_all_write.shape} ***********"
#     )

#     split_data_pred_all_write.rename(
#         columns={"PREDICTED_PROBABILITY": "PROBABILITY_OF_RISKY"}, inplace=True
#     )

#     truncate_table(
#         "dod_results_ab_test_full_write_monitoring", "cm_ac_vanilla_v1".lower(), cur
#     )
#     write_to_snowflake(
#         split_data_pred_all_write,
#         "dod_results_ab_test_full_write_monitoring",
#         "cm_ac_vanilla_v1".lower(),
#     )

#     split_data_pred_risky = split_data_pred.loc[
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 1, :
#     ]
#     split_data_pred_risky_share = split_data_pred_risky.drop(
#         columns=["PREDICTED_BINARY_VERDICT", "PREDICTED_PROBABILITY"]
#     )

#     split_data_pred_risky_share["DB_UPDATED_AT"] = split_data_pred_risky_share[
#         "DB_UPDATED_AT"
#     ].astype("str")

#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred.shape} ***********"
#     )
#     print(
#         f"*********** predicted split data for DS mode  : {split_data_pred.head()} ***********"
#     )
#     print(
#         f"*********** Risky predicted split data for DS mode shape  : {split_data_pred_risky_share.shape} ***********"
#     )

#     split_data_pred_risky_share["RUN_TIMESTAMP"] = pd.to_datetime(datetime.now())
#     split_data_pred_risky_share["RUN_TIMESTAMP"] = split_data_pred_risky_share[
#         "RUN_TIMESTAMP"
#     ].astype("str")

#     truncate_table("dod_results_ab_test", "cm_ac_vanilla_v1".lower(), cur)
#     write_to_snowflake(
#         split_data_pred_risky_share, "dod_results_ab_test", "cm_ac_vanilla_v1".lower()
#     )

#     # writing to common combination table
#     trigger_cmd(gd.write_to_combination_table, conn)

#     # writing to master table for monitoring
#     trigger_cmd(gd.write_split_full_data, conn)

#     # merge back to master predicted result table for 65% predicted risky of 50% total
#     trigger_cmd(gd.merge_predicted_split_data, conn)

#     # merging prediction to final table
#     trigger_cmd(gd.merge_predicted_data, conn)

#     cur.close()
#     conn.close()


## logic testing 
# def model_run():
#     # start_date = datetime.now().strftime("%Y-%m-%d")
#     # end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
#     start_date = '2023-07-26'
#     end_date = '2023-07-27'

#     # get data
#     data = get_raw_data(start_date, end_date, cur, query)

#     print(f"*********** full data shape : {data.shape} ***********")

#     change_list = ["LOAN_TYPE", "USER_TYPE", "VERIFICATION_TYPE", "PINCODE_RISK_FLAG"]

#     def change_list_fn(data):
#         for column in change_list:
#             if gd.var_type(column) == "Categorical":
#                 # print(column)
#                 data[column] = data[column].fillna("MISSING").apply(lambda x: x.upper())
#         return data

#     data = change_list_fn(data)

#     def encode_categ(data):
#         category_loan_type_mapping = {"FRESH": 0, "RENEWAL": 1, "MISSING": 0}
#         data["LOAN_TYPE_ENCODED"] = data["LOAN_TYPE"].map(category_loan_type_mapping)

#         category_user_type_mapping = {"KB USER": 1, "NON-KB USER": 0, "MISSING": 0}
#         data["USER_TYPE_ENCODED"] = data["USER_TYPE"].map(category_user_type_mapping)

#         category_verification_type_mapping = {
#             "PHYSICALLY VERIFIED": 2,
#             "VIDEO VERIFIED": 1,
#             "MISSING": 0,
#         }
#         data["VERIFICATION_TYPE_ENCODED"] = data["VERIFICATION_TYPE"].map(
#             category_verification_type_mapping
#         )

#         category_riskpincode_type_mapping = {
#             "HIGH": 2,
#             "MEDIUM": 1,
#             "LOW": 0,
#             "MISSING": 0,
#         }
#         data["PINCODE_RISK_FLAG_ENCODED"] = data["PINCODE_RISK_FLAG"].map(
#             category_riskpincode_type_mapping
#         )

#         return data

#     data = encode_categ(data)

#     # pick only selected features utilized in model
#     feature_list = pd.read_csv(
#         read_file(
#             uf.s3_bucket, uf.model_path + "XGBoost_feature_list_100_prev_v3_2.csv"
#         )
#     )["variables"]
#     # select only cols used in model
#     data_filter = data[list(feature_list)]

#     # remove_inconsistent_values(data)
#     data_filter = remove_inconsistent_values(data_filter, feature_list)

#     data_filter = gd.missing_ind_convert_num(data_filter)
#     # unpickle cm_ac_vanilla_v1 model
#     model = pickle.loads(
#         uf.s3.Bucket(uf.s3_bucket)
#         .Object(f"{uf.model_path}Vanilla_model_v3_1_filter.pkl")
#         .get()["Body"]
#         .read()
#     )

#     # run predictions and store the result
#     data["Predicted_probability"] = model.predict_proba(data_filter)[:, 1]

#     # finding threshold value for 35% prediction coverage
#     threshold_value = uf.find_threshold(data, uf.cut_off, "DS")

#     # converting threshold cut to binary
#     data["Predicted_binary_verdict"] = np.where(
#         data["Predicted_probability"] >= threshold_value, 1, 0
#     )

#     # BAD FLAG
#     # data['BAD_FLAG'] = np.where(data['NEXT_DAY_DPD']==2,1,0)

#     # data['BAD_FLAG'].value_counts()

#     # store only relevant info:
#     data_write = data[
#         list(feature_list)
#         + [
#             "LOAN_ID",
#             "USER_ID",
#             "FULL_DATE",
#             "Predicted_probability",
#             "Predicted_binary_verdict",
#             "AUTO_CURE_FLAG",
#         ]
#     ]

#     # precision = precision_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
#     # Calculate recall
#     # recall = recall_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
#     # print("Precision Val data  {:.4f}%".format(precision*100))
#     # print("Recall Val data {:.4f}%".format(recall*100))

#     # fpr, tpr, thresholds = roc_curve(data_write['BAD_FLAG'], data_write["Predicted_probability"])
#     # roc_auc = auc(fpr, tpr)
#     # GINI_Val = (2 * roc_auc) - 1
#     # print(f'Gini on Val data with XGBoost: {GINI_Val}')

#     # write back to snowflake:
#     truncate_table("dod_results", "cm_ac_vanilla_v1".lower(), cur)
#     # write temp DoD table:
#     write_to_snowflake(data_write, "dod_results", "cm_ac_vanilla_v1".lower())

#     split_data = get_raw_data(start_date, end_date, cur, gd.get_split_data)
#     print(f"*********** split data for DS mode shape : {split_data.shape} ***********")

#     exclude = [
#         "RISK_FLAG",
#         "ONE_DPD_TELE",
#         "FINAL_RISK_FLAG",
#         "RISK_RANK",
#         "PERCENTILE",
#         "ROW_NUM",
#         "MODEL",
#         "TOTAL_ROWS",
#     ]

#     split_data_1 = split_data.drop(columns=exclude)
#     data_write.rename(columns={"FULL_DATE": "UPDATE_DT"}, inplace=True)
#     # converting all columns to UPPER CASE
#     data_write.columns = data_write.columns.str.upper()

#     split_data_pred = split_data_1.merge(
#         data_write[
#             [
#                 "LOAN_ID",
#                 "UPDATE_DT",
#                 "PREDICTED_PROBABILITY",
#                 "PREDICTED_BINARY_VERDICT",
#             ]
#         ],
#         how="inner",
#         on=["LOAN_ID", "UPDATE_DT"],
#     )

#     split_data_pred_all = copy.copy(split_data_pred)
#     split_data_pred_all["GROUP_NAME"] = "Treatment"

#     split_data_pred_all["RISK_FLAG"] = np.where(
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 0, "Non-Risky", "Risky"
#     )
#     split_data_pred_all_write = split_data_pred_all[
#         [
#             "UPDATE_DT",
#             "LOAN_ID",
#             "TODAYS_EDI",
#             "PREDICTED_PROBABILITY",
#             "RISK_FLAG",
#             "GROUP_NAME",
#         ]
#     ]

#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred_all_write.shape} ***********"
#     )

#     split_data_pred_all_write.rename(
#         columns={"PREDICTED_PROBABILITY": "PROBABILITY_OF_RISKY"}, inplace=True
#     )

#     truncate_table(
#         "dod_results_ab_test_full_write_monitoring", "cm_ac_vanilla_v1".lower(), cur
#     )
#     write_to_snowflake(
#         split_data_pred_all_write,
#         "dod_results_ab_test_full_write_monitoring",
#         "cm_ac_vanilla_v1".lower(),
#     )

#     split_data_pred_risky = split_data_pred.loc[
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 1, :
#     ]
#     split_data_pred_risky_share = split_data_pred_risky.drop(
#         columns=["PREDICTED_BINARY_VERDICT", "PREDICTED_PROBABILITY"]
#     )

#     split_data_pred_risky_share["DB_UPDATED_AT"] = split_data_pred_risky_share[
#         "DB_UPDATED_AT"
#     ].astype("str")

#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred.shape} ***********"
#     )
#     print(
#         f"*********** predicted split data for DS mode  : {split_data_pred.head()} ***********"
#     )
#     print(
#         f"*********** Risky predicted split data for DS mode shape  : {split_data_pred_risky_share.shape} ***********"
#     )

#     split_data_pred_risky_share["RUN_TIMESTAMP"] = pd.to_datetime(datetime.now())
#     split_data_pred_risky_share["RUN_TIMESTAMP"] = split_data_pred_risky_share[
#         "RUN_TIMESTAMP"
#     ].astype("str")

#     truncate_table("dod_results_ab_test", "cm_ac_vanilla_v1".lower(), cur)
#     write_to_snowflake(
#         split_data_pred_risky_share, "dod_results_ab_test", "cm_ac_vanilla_v1".lower()
#     )

#     # writing to common combination table
#     trigger_cmd(gd.write_to_combination_table, conn)

#     # writing to master table for monitoring
#     trigger_cmd(gd.write_split_full_data, conn)

#     # merge back to master predicted result table for 65% predicted risky of 50% total
#     trigger_cmd(gd.merge_predicted_split_data, conn)

#     # merging prediction to final table
#     trigger_cmd(gd.merge_predicted_data, conn)

#     cur.close()
#     conn.close()


## model v3 - expanded data set and increased features
# def model_run():
#     start_date = datetime.now().strftime("%Y-%m-%d")
#     end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

#     # get data
#     data = get_raw_data(start_date, end_date, cur, gd.query)

#     print(f"*********** full data shape : {data.shape} ***********")

#     change_list = ['LOAN_TYPE',
#         'USER_TYPE',
#         'VERIFICATION_TYPE',
#         'PINCODE_RISK_FLAG'
#         ]
    
#     def change_list_fn(data):
#         for column in change_list:
#             if gd.var_type(column)=="Categorical":
#                 # print(column)
#                 data[column] = data[column].fillna('MISSING').apply(lambda x: x.upper())
#         return data 

#     data = change_list_fn(data)

#     def encode_categ(data):
#         category_loan_type_mapping = {'FRESH': 0, 'RENEWAL': 1, 'MISSING': 0}
#         data['LOAN_TYPE_ENCODED'] = data['LOAN_TYPE'].map(category_loan_type_mapping)

#         category_user_type_mapping = {'KB USER': 1, 'NON-KB USER': 0, 'MISSING': 0}
#         data['USER_TYPE_ENCODED'] = data['USER_TYPE'].map(category_user_type_mapping)

#         category_verification_type_mapping = {'PHYSICALLY VERIFIED': 2, 'VIDEO VERIFIED': 1, 'MISSING': 0}
#         data['VERIFICATION_TYPE_ENCODED'] = data['VERIFICATION_TYPE'].map(category_verification_type_mapping)

#         category_riskpincode_type_mapping = {'HIGH': 2, 'MEDIUM': 1, 'LOW': 0, 'MISSING': 0}
#         data['PINCODE_RISK_FLAG_ENCODED'] = data['PINCODE_RISK_FLAG'].map(category_riskpincode_type_mapping)

#         return data

#     data = encode_categ(data)

#     # pick only selected features utilized in model
#     feature_list = pd.read_csv(
#         read_file(uf.s3_bucket, uf.model_path + "XGBoost_feature_list_100_prev_v3_2.csv")
#     )["variables"]
#     # select only cols used in model
#     data_filter = data[list(feature_list)]

#     # remove_inconsistent_values(data)
#     data_filter = remove_inconsistent_values(data_filter, feature_list)

#     data_filter = gd.missing_ind_convert_num(data_filter)
#     # unpickle cm_ac_vanilla_v1 model
#     model = pickle.loads(
#         uf.s3.Bucket(uf.s3_bucket)
#         .Object(f"{uf.model_path}Vanilla_model_v3_1_filter.pkl")
#         .get()["Body"]
#         .read()
#     )

#     # run predictions and store the result
#     data["Predicted_probability"] = model.predict_proba(data_filter)[:, 1]

#     # finding threshold value for 35% prediction coverage
#     threshold_value = uf.find_threshold(data, uf.cut_off, "DS")

#     # converting threshold cut to binary
#     data["Predicted_binary_verdict"] = np.where(
#         data["Predicted_probability"] >= threshold_value, 1, 0
#     )

#     # BAD FLAG
#     # data['BAD_FLAG'] = np.where(data['NEXT_DAY_DPD']==2,1,0)

#     # data['BAD_FLAG'].value_counts()

#     # store only relevant info:
#     data_write = data[
#         list(feature_list)
#         + [
#             "LOAN_ID",
#             "USER_ID",
#             "FULL_DATE",
#             "Predicted_probability",
#             "Predicted_binary_verdict",
#             "AUTO_CURE_FLAG"
#         ]
#     ]

#     # precision = precision_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
#     # Calculate recall
#     # recall = recall_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
#     # print("Precision Val data  {:.4f}%".format(precision*100))
#     # print("Recall Val data {:.4f}%".format(recall*100))

#     # fpr, tpr, thresholds = roc_curve(data_write['BAD_FLAG'], data_write["Predicted_probability"])
#     # roc_auc = auc(fpr, tpr)
#     # GINI_Val = (2 * roc_auc) - 1
#     # print(f'Gini on Val data with XGBoost: {GINI_Val}')

#     # write back to snowflake:
#     truncate_table("dod_results", "cm_ac_vanilla_v1_temp".lower(), cur)
#     # write temp DoD table:
#     write_to_snowflake(data_write, "dod_results", "cm_ac_vanilla_v1_temp".lower())    
    
#     split_data = get_raw_data(start_date, end_date, cur, gd.get_split_data)
#     print(f"*********** split data for DS mode shape : {split_data.shape} ***********")

#     exclude = [
#         "RISK_FLAG",
#         "ONE_DPD_TELE",
#         "FINAL_RISK_FLAG",
#         "RISK_RANK",
#         "PERCENTILE",
#         "ROW_NUM",
#         "MODEL",
#         "TOTAL_ROWS"
#     ]

#     split_data_1 = split_data.drop(columns=exclude)
#     data_write.rename(columns={"FULL_DATE": "UPDATE_DT"}, inplace=True)
#     # converting all columns to UPPER CASE
#     data_write.columns = data_write.columns.str.upper()

#     split_data_pred = split_data_1.merge(
#         data_write[
#             [
#                 "LOAN_ID",
#                 "UPDATE_DT",
#                 "PREDICTED_PROBABILITY",
#                 "PREDICTED_BINARY_VERDICT",
#             ]
#         ],
#         how="inner",
#         on=["LOAN_ID", "UPDATE_DT"],
#     )

#     split_data_pred_all = copy.copy(split_data_pred)
#     split_data_pred_all["GROUP_NAME"] = "Treatment"

#     split_data_pred_all["RISK_FLAG"] = np.where(
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 0, "Non-Risky", "Risky"
#     )
#     split_data_pred_all_write = split_data_pred_all[
#         [
#             "UPDATE_DT",
#             "LOAN_ID",
#             "TODAYS_EDI",
#             "PREDICTED_PROBABILITY",
#             "RISK_FLAG",
#             "GROUP_NAME",
#         ]
#     ]

#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred_all_write.shape} ***********"
#     )

#     split_data_pred_all_write.rename(
#         columns={"PREDICTED_PROBABILITY": "PROBABILITY_OF_RISKY"}, inplace=True
#     )

#     truncate_table(
#         "dod_results_ab_test_full_write_monitoring", "cm_ac_vanilla_v1".lower(), cur
#     )
#     write_to_snowflake(
#         split_data_pred_all_write,
#         "dod_results_ab_test_full_write_monitoring",
#         "cm_ac_vanilla_v1".lower(),
#     )

#     split_data_pred_risky = split_data_pred.loc[
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 1, :
#     ]
#     split_data_pred_risky_share = split_data_pred_risky.drop(
#         columns=["PREDICTED_BINARY_VERDICT", "PREDICTED_PROBABILITY"]
#     )

#     split_data_pred_risky_share["DB_UPDATED_AT"] = split_data_pred_risky_share[
#         "DB_UPDATED_AT"
#     ].astype("str")

#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred.shape} ***********"
#     )
#     print(
#         f"*********** predicted split data for DS mode  : {split_data_pred.head()} ***********"
#     )
#     print(
#         f"*********** Risky predicted split data for DS mode shape  : {split_data_pred_risky_share.shape} ***********"
#     )

#     split_data_pred_risky_share["RUN_TIMESTAMP"] = pd.to_datetime(datetime.now())
#     split_data_pred_risky_share["RUN_TIMESTAMP"] = split_data_pred_risky_share[
#         "RUN_TIMESTAMP"
#     ].astype("str")

#     truncate_table("dod_results_ab_test", "cm_ac_vanilla_v1".lower(), cur)
#     write_to_snowflake(
#         split_data_pred_risky_share, "dod_results_ab_test", "cm_ac_vanilla_v1".lower()
#     )

#     # writing to common combination table
#     trigger_cmd(gd.write_to_combination_table, conn)

#     # writing to master table for monitoring
#     trigger_cmd(gd.write_split_full_data, conn)

#     # merge back to master predicted result table for 65% predicted risky of 50% total
#     trigger_cmd(gd.merge_predicted_split_data, conn)

#     # merging prediction to final table
#     trigger_cmd(gd.merge_predicted_data, conn)

#     cur.close()
#     conn.close()

## model v1 - 3 expanded train set
# def model_run():
#     start_date = datetime.now().strftime("%Y-%m-%d")
#     end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

#     # get data
#     data = get_raw_data(start_date, end_date, cur, gd.query)

#     print(f"*********** full data shape : {data.shape} ***********")

#     # pick only selected features utilized in model
#     feature_list = pd.read_csv(
#         read_file(uf.s3_bucket, uf.model_path + "XGBoost_feature_list_100_prev_new_expand.csv")
#     )["variables"]
#     # select only cols used in model
#     data_filter = data[list(feature_list)]

#     # remove_inconsistent_values(data)
#     data_filter = remove_inconsistent_values(data_filter, feature_list)

#     data_filter = gd.missing_ind_convert_num(data_filter)
#     # unpickle cm_ac_vanilla_v1 model
#     model = pickle.loads(
#         uf.s3.Bucket(uf.s3_bucket)
#         .Object(f"{uf.model_path}Vanilla_model_v1_app_1_expand.pkl")
#         .get()["Body"]
#         .read()
#     )

#     # run predictions and store the result
#     data["Predicted_probability"] = model.predict_proba(data_filter)[:, 1]

#     # finding threshold value for 35% prediction coverage
#     threshold_value = uf.find_threshold(data, uf.cut_off, "DS")

#     # converting threshold cut to binary
#     data["Predicted_binary_verdict"] = np.where(
#         data["Predicted_probability"] >= threshold_value, 1, 0
#     )

#     # BAD FLAG
#     # data['BAD_FLAG'] = np.where(data['NEXT_DAY_DPD']==2,1,0)

#     # data['BAD_FLAG'].value_counts()

#     # store only relevant info:
#     data_write = data[
#         list(feature_list)
#         + [
#             "LOAN_ID",
#             "USER_ID",
#             "FULL_DATE",
#             "Predicted_probability",
#             "Predicted_binary_verdict",
#             "AUTO_CURE_FLAG"
#         ]
#     ]

#     # precision = precision_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
#     # # Calculate recall
#     # recall = recall_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
#     # print("Precision Val data  {:.4f}%".format(precision*100))
#     # print("Recall Val data {:.4f}%".format(recall*100))

#     # fpr, tpr, thresholds = roc_curve(data_write['BAD_FLAG'], data_write["Predicted_probability"])
#     # roc_auc = auc(fpr, tpr)
#     # GINI_Val = (2 * roc_auc) - 1
#     # print(f'Gini on Val data with XGBoost: {GINI_Val}')

#     # write back to snowflake:
#     truncate_table("dod_results", "cm_ac_vanilla_v1".lower(), cur)
#     # write temp DoD table:
#     write_to_snowflake(data_write, "dod_results", "cm_ac_vanilla_v1".lower())

#     split_data = get_raw_data(start_date, end_date, cur, gd.get_split_data)
#     print(f"*********** split data for DS mode shape : {split_data.shape} ***********")

#     exclude = [
#         "RISK_FLAG",
#         "ONE_DPD_TELE",
#         "FINAL_RISK_FLAG",
#         "RISK_RANK",
#         "PERCENTILE",
#         "ROW_NUM",
#         "MODEL",
#         "TOTAL_ROWS"
#     ]

#     split_data_1 = split_data.drop(columns=exclude)
#     data_write.rename(columns={"FULL_DATE": "UPDATE_DT"}, inplace=True)
#     # converting all columns to UPPER CASE
#     data_write.columns = data_write.columns.str.upper()

#     split_data_pred = split_data_1.merge(
#         data_write[
#             [
#                 "LOAN_ID",
#                 "UPDATE_DT",
#                 "PREDICTED_PROBABILITY",
#                 "PREDICTED_BINARY_VERDICT",
#             ]
#         ],
#         how="inner",
#         on=["LOAN_ID", "UPDATE_DT"],
#     )

#     split_data_pred_all = copy.copy(split_data_pred)
#     split_data_pred_all["GROUP_NAME"] = "Treatment"

#     split_data_pred_all["RISK_FLAG"] = np.where(
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 0, "Non-Risky", "Risky"
#     )
#     split_data_pred_all_write = split_data_pred_all[
#         [
#             "UPDATE_DT",
#             "LOAN_ID",
#             "TODAYS_EDI",
#             "PREDICTED_PROBABILITY",
#             "RISK_FLAG",
#             "GROUP_NAME",
#         ]
#     ]

#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred_all_write.shape} ***********"
#     )

#     split_data_pred_all_write.rename(
#         columns={"PREDICTED_PROBABILITY": "PROBABILITY_OF_RISKY"}, inplace=True
#     )

#     truncate_table(
#         "dod_results_ab_test_full_write_monitoring", "cm_ac_vanilla_v1".lower(), cur
#     )
#     write_to_snowflake(
#         split_data_pred_all_write,
#         "dod_results_ab_test_full_write_monitoring",
#         "cm_ac_vanilla_v1".lower(),
#     )

#     split_data_pred_risky = split_data_pred.loc[
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 1, :
#     ]
#     split_data_pred_risky_share = split_data_pred_risky.drop(
#         columns=["PREDICTED_BINARY_VERDICT", "PREDICTED_PROBABILITY"]
#     )

#     split_data_pred_risky_share["DB_UPDATED_AT"] = split_data_pred_risky_share[
#         "DB_UPDATED_AT"
#     ].astype("str")

#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred.shape} ***********"
#     )
#     print(
#         f"*********** predicted split data for DS mode  : {split_data_pred.head()} ***********"
#     )
#     print(
#         f"*********** Risky predicted split data for DS mode shape  : {split_data_pred_risky_share.shape} ***********"
#     )

#     split_data_pred_risky_share["RUN_TIMESTAMP"] = pd.to_datetime(datetime.now())
#     split_data_pred_risky_share["RUN_TIMESTAMP"] = split_data_pred_risky_share[
#         "RUN_TIMESTAMP"
#     ].astype("str")

#     truncate_table("dod_results_ab_test", "cm_ac_vanilla_v1".lower(), cur)
#     write_to_snowflake(
#         split_data_pred_risky_share, "dod_results_ab_test", "cm_ac_vanilla_v1".lower()
#     )

#     # writing to common combination table
#     trigger_cmd(gd.write_to_combination_table, conn)

#     # writing to master table for monitoring
#     trigger_cmd(gd.write_split_full_data, conn)

#     # merging prediction to final table
#     trigger_cmd(gd.merge_predicted_data, conn)  

#     # merge back to master predicted result table for 65% predicted risky of 50% total
#     trigger_cmd(gd.merge_predicted_split_data, conn)

#     cur.close()
#     conn.close()





    # start_date = datetime.now().strftime("%Y-%m-%d")
    # end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # # get data
    # data = get_raw_data(start_date, end_date, cur, gd.query)

    # print(f"*********** full data shape : {data.shape} ***********")

    # # pick only selected features utilized in model
    # feature_list = pd.read_csv(
    #     read_file(uf.s3_bucket, uf.model_path + "XGBoost_feature_list_100_prev_new_expand.csv")
    # )["variables"]
    # # select only cols used in model
    # data_filter = data[list(feature_list)]

    # # remove_inconsistent_values(data)
    # data_filter = remove_inconsistent_values(data_filter, feature_list)

    # data_filter = gd.missing_ind_convert_num(data_filter)
    # # unpickle cm_ac_vanilla_v1 model
    # model = pickle.loads(
    #     uf.s3.Bucket(uf.s3_bucket)
    #     .Object(f"{uf.model_path}Vanilla_model_v1_app_1_expand.pkl")
    #     .get()["Body"]
    #     .read()
    # )

    # # run predictions and store the result
    # data["Predicted_probability"] = model.predict_proba(data_filter)[:, 1]

    # # finding threshold value for 35% prediction coverage
    # threshold_value = uf.find_threshold(data, uf.cut_off, "DS")

    # # converting threshold cut to binary
    # data["Predicted_binary_verdict"] = np.where(
    #     data["Predicted_probability"] >= threshold_value, 1, 0
    # )

    # # BAD FLAG
    # # data['BAD_FLAG'] = np.where(data['NEXT_DAY_DPD']==2,1,0)

    # # data['BAD_FLAG'].value_counts()

    # # store only relevant info:
    # data_write = data[
    #     list(feature_list)
    #     + [
    #         "LOAN_ID",
    #         "USER_ID",
    #         "FULL_DATE",
    #         "Predicted_probability",
    #         "Predicted_binary_verdict",
    #         "AUTO_CURE_FLAG"
    #     ]
    # ]

    # precision = precision_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
    # # Calculate recall
    # recall = recall_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
    # print("Precision Val data  {:.4f}%".format(precision*100))
    # print("Recall Val data {:.4f}%".format(recall*100))

    # fpr, tpr, thresholds = roc_curve(data_write['BAD_FLAG'], data_write["Predicted_probability"])
    # roc_auc = auc(fpr, tpr)
    # GINI_Val = (2 * roc_auc) - 1
    # print(f'Gini on Val data with XGBoost: {GINI_Val}')

    # # write back to snowflake:
    # truncate_table("dod_results", "cm_ac_vanilla_v1_temp".lower(), cur)
    # # write temp DoD table:
    # write_to_snowflake(data_write, "dod_results", "cm_ac_vanilla_v1_temp".lower())

    # split_data = get_raw_data(start_date, end_date, cur, gd.get_split_data)
    # print(f"*********** split data for DS mode shape : {split_data.shape} ***********")

    # exclude = [
    #     "RISK_FLAG",
    #     "ONE_DPD_TELE",
    #     "FINAL_RISK_FLAG",
    #     "RISK_RANK",
    #     "PERCENTILE",
    #     "ROW_NUM",
    #     "MODEL",
    #     "TOTAL_ROWS"
    # ]

    # split_data_1 = split_data.drop(columns=exclude)
    # data_write.rename(columns={"FULL_DATE": "UPDATE_DT"}, inplace=True)
    # # converting all columns to UPPER CASE
    # data_write.columns = data_write.columns.str.upper()

    # split_data_pred = split_data_1.merge(
    #     data_write[
    #         [
    #             "LOAN_ID",
    #             "UPDATE_DT",
    #             "PREDICTED_PROBABILITY",
    #             "PREDICTED_BINARY_VERDICT",
    #         ]
    #     ],
    #     how="inner",
    #     on=["LOAN_ID", "UPDATE_DT"],
    # )

    # split_data_pred_all = copy.copy(split_data_pred)
    # split_data_pred_all["GROUP_NAME"] = "Treatment"

    # split_data_pred_all["RISK_FLAG"] = np.where(
    #     split_data_pred["PREDICTED_BINARY_VERDICT"] == 0, "Non-Risky", "Risky"
    # )
    # split_data_pred_all_write = split_data_pred_all[
    #     [
    #         "UPDATE_DT",
    #         "LOAN_ID",
    #         "TODAYS_EDI",
    #         "PREDICTED_PROBABILITY",
    #         "RISK_FLAG",
    #         "GROUP_NAME",
    #     ]
    # ]

    # print(
    #     f"*********** predicted split data for DS mode shape  : {split_data_pred_all_write.shape} ***********"
    # )

    # split_data_pred_all_write.rename(
    #     columns={"PREDICTED_PROBABILITY": "PROBABILITY_OF_RISKY"}, inplace=True
    # )

    # truncate_table(
    #     "dod_results_ab_test_full_write_monitoring", "cm_ac_vanilla_v1".lower(), cur
    # )
    # write_to_snowflake(
    #     split_data_pred_all_write,
    #     "dod_results_ab_test_full_write_monitoring",
    #     "cm_ac_vanilla_v1".lower(),
    # )

    # split_data_pred_risky = split_data_pred.loc[
    #     split_data_pred["PREDICTED_BINARY_VERDICT"] == 1, :
    # ]
    # split_data_pred_risky_share = split_data_pred_risky.drop(
    #     columns=["PREDICTED_BINARY_VERDICT", "PREDICTED_PROBABILITY"]
    # )

    # split_data_pred_risky_share["DB_UPDATED_AT"] = split_data_pred_risky_share[
    #     "DB_UPDATED_AT"
    # ].astype("str")

    # print(
    #     f"*********** predicted split data for DS mode shape  : {split_data_pred.shape} ***********"
    # )
    # print(
    #     f"*********** predicted split data for DS mode  : {split_data_pred.head()} ***********"
    # )
    # print(
    #     f"*********** Risky predicted split data for DS mode shape  : {split_data_pred_risky_share.shape} ***********"
    # )

    # split_data_pred_risky_share["RUN_TIMESTAMP"] = pd.to_datetime(datetime.now())
    # split_data_pred_risky_share["RUN_TIMESTAMP"] = split_data_pred_risky_share[
    #     "RUN_TIMESTAMP"
    # ].astype("str")

    # truncate_table("dod_results_ab_test", "cm_ac_vanilla_v1".lower(), cur)
    # write_to_snowflake(
    #     split_data_pred_risky_share, "dod_results_ab_test", "cm_ac_vanilla_v1".lower()
    # )

    # # writing to common combination table
    # trigger_cmd(gd.write_to_combination_table, conn)

    # # writing to master table for monitoring
    # trigger_cmd(gd.write_split_full_data, conn)

    # # merging prediction to final table
    # trigger_cmd(gd.merge_predicted_data, conn)

    # # merge back to master predicted result table for 65% predicted risky of 50% total
    # trigger_cmd(gd.merge_predicted_split_data, conn)

    # cur.close()
    # conn.close()

# model_run()




## Model v1 - 2 stage repo
# def model_run():
#     start_date = datetime.now().strftime("%Y-%m-%d")
#     end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

#     # get data
#     data = get_raw_data(start_date, end_date, cur, gd.query)

#     print(f"*********** full data shape : {data.shape} ***********")

#     # pick only selected features utilized in model
#     feature_list = pd.read_csv(
#         read_file(uf.s3_bucket, uf.model_path + "XGBoost_feature_list_100_prev_new_app.csv")
#     )["variables"]
#     # select only cols used in model
#     data_filter = data[list(feature_list)]
#     # remove_inconsistent_values(data)
#     data_filter = remove_inconsistent_values(data_filter, feature_list)

#     data_filter = gd.missing_ind_convert_num(data_filter)
#     # unpickle cm_ac_vanilla_v1 model
#     model = pickle.loads(
#         uf.s3.Bucket(uf.s3_bucket)
#         .Object(f"{uf.model_path}Vanilla_model_v1_app_1.pkl")
#         .get()["Body"]
#         .read()
#     )

#     # run predictions and store the result
#     data["Predicted_probability"] = model.predict_proba(data_filter)[:, 1]

#     # finding threshold value for 35% prediction coverage
#     threshold_value = uf.find_threshold(data, uf.cut_off, "DS")

#     # converting threshold cut to binary
#     data["Predicted_binary_verdict"] = np.where(
#         data["Predicted_probability"] >= threshold_value, 1, 0
#     )

#     # store only relevant info:
#     data_write = data[
#         list(feature_list)
#         + [
#             "LOAN_ID",
#             "USER_ID",
#             "FULL_DATE",
#             "Predicted_probability",
#             "Predicted_binary_verdict",
#             "AUTO_CURE_FLAG"
#         ]
#     ]

#     # precision = precision_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
#     # # Calculate recall
#     # recall = recall_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
#     # print("Precision Val data  {:.4f}%".format(precision*100))
#     # print("Recall Val data {:.4f}%".format(recall*100))

#     # fpr, tpr, thresholds = roc_curve(data_write['BAD_FLAG'], data_write["Predicted_probability"])
#     # roc_auc = auc(fpr, tpr)
#     # GINI_Val = (2 * roc_auc) - 1
#     # print(f'Gini on Val data with XGBoost: {GINI_Val}')

#     # write back to snowflake:
#     truncate_table("dod_results", "cm_ac_vanilla_v1".lower(), cur)
#     # write temp DoD table:
#     write_to_snowflake(data_write, "dod_results", "cm_ac_vanilla_v1".lower())

#     split_data = get_raw_data(start_date, end_date, cur, gd.get_split_data)
#     print(f"*********** split data for DS mode shape : {split_data.shape} ***********")

#     exclude = [
#         "RISK_FLAG",
#         "ONE_DPD_TELE",
#         "FINAL_RISK_FLAG",
#         "RISK_RANK",
#         "PERCENTILE",
#         "ROW_NUM",
#         "MODEL",
#         "TOTAL_ROWS"
#     ]

#     split_data_1 = split_data.drop(columns=exclude)
#     data_write.rename(columns={"FULL_DATE": "UPDATE_DT"}, inplace=True)
#     # converting all columns to UPPER CASE
#     data_write.columns = data_write.columns.str.upper()

#     split_data_pred = split_data_1.merge(
#         data_write[
#             [
#                 "LOAN_ID",
#                 "UPDATE_DT",
#                 "PREDICTED_PROBABILITY",
#                 "PREDICTED_BINARY_VERDICT",
#             ]
#         ],
#         how="inner",
#         on=["LOAN_ID", "UPDATE_DT"],
#     )

#     split_data_pred_all = copy.copy(split_data_pred)
#     split_data_pred_all["GROUP_NAME"] = "Treatment"

#     split_data_pred_all["RISK_FLAG"] = np.where(
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 0, "Non-Risky", "Risky"
#     )
#     split_data_pred_all_write = split_data_pred_all[
#         [
#             "UPDATE_DT",
#             "LOAN_ID",
#             "TODAYS_EDI",
#             "PREDICTED_PROBABILITY",
#             "RISK_FLAG",
#             "GROUP_NAME",
#         ]
#     ]

#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred_all_write.shape} ***********"
#     )

#     split_data_pred_all_write.rename(
#         columns={"PREDICTED_PROBABILITY": "PROBABILITY_OF_RISKY"}, inplace=True
#     )

#     truncate_table(
#         "dod_results_ab_test_full_write_monitoring", "cm_ac_vanilla_v1".lower(), cur
#     )
#     write_to_snowflake(
#         split_data_pred_all_write,
#         "dod_results_ab_test_full_write_monitoring",
#         "cm_ac_vanilla_v1".lower(),
#     )

#     split_data_pred_risky = split_data_pred.loc[
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 1, :
#     ]
#     split_data_pred_risky_share = split_data_pred_risky.drop(
#         columns=["PREDICTED_BINARY_VERDICT", "PREDICTED_PROBABILITY"]
#     )

#     split_data_pred_risky_share["DB_UPDATED_AT"] = split_data_pred_risky_share[
#         "DB_UPDATED_AT"
#     ].astype("str")

#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred.shape} ***********"
#     )
#     print(
#         f"*********** predicted split data for DS mode  : {split_data_pred.head()} ***********"
#     )
#     print(
#         f"*********** Risky predicted split data for DS mode shape  : {split_data_pred_risky_share.shape} ***********"
#     )

#     split_data_pred_risky_share["RUN_TIMESTAMP"] = pd.to_datetime(datetime.now())
#     split_data_pred_risky_share["RUN_TIMESTAMP"] = split_data_pred_risky_share[
#         "RUN_TIMESTAMP"
#     ].astype("str")

#     truncate_table("dod_results_ab_test", "cm_ac_vanilla_v1".lower(), cur)
#     write_to_snowflake(
#         split_data_pred_risky_share, "dod_results_ab_test", "cm_ac_vanilla_v1".lower()
#     )

#     # writing to common combination table
#     trigger_cmd(gd.write_to_combination_table, conn)

#     # writing to master table for monitoring
#     trigger_cmd(gd.write_split_full_data, conn)

#     # merging prediction to final table
#     trigger_cmd(gd.merge_predicted_data, conn)

#     # merge back to master predicted result table for 65% predicted risky of 50% total
#     trigger_cmd(gd.merge_predicted_split_data, conn)

#     cur.close()
#     conn.close()





## Model v1 - 2 changed approach

# def model_run():
#     start_date = datetime.now().strftime("%Y-%m-%d")
#     end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

#     # get data
#     data = get_raw_data(start_date, end_date, cur, gd.query)

#     data.shape

#     # pick only selected features utilized in model
#     feature_list = pd.read_csv(
#         read_file(uf.s3_bucket, uf.model_path + "XGBoost_feature_list_100_prev_new_app.csv")
#     )["variables"]
#     # select only cols used in model
#     data_filter = data[list(feature_list)]
#     # remove_inconsistent_values(data)
#     data_filter = remove_inconsistent_values(data_filter, feature_list)

#     data_filter = gd.missing_ind_convert_num(data_filter)
#     # unpickle cm_ac_vanilla_v1 model
#     model = pickle.loads(
#         uf.s3.Bucket(uf.s3_bucket)
#         .Object(f"{uf.model_path}Vanilla_model_v1_app_1.pkl")
#         .get()["Body"]
#         .read()
#     )

#     # run predictions and store the result
#     data["Predicted_probability"] = model.predict_proba(data_filter)[:, 1]

#     # finding threshold value for 35% prediction coverage
#     threshold_value = uf.find_threshold(data, uf.cut_off, "DS")

#     # converting threshold cut to binary
#     data["Predicted_binary_verdict"] = np.where(
#         data["Predicted_probability"] >= threshold_value, 1, 0
#     )

#     # store only relevant info:
#     data_write = data[
#         list(feature_list)
#         + [
#             "LOAN_ID",
#             "USER_ID",
#             "FULL_DATE",
#             "Predicted_probability",
#             "Predicted_binary_verdict",
#             "AUTO_CURE_FLAG"
#         ]
#     ]

#     # precision = precision_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
#     # # Calculate recall
#     # recall = recall_score(data_write['BAD_FLAG'], data_write["Predicted_binary_verdict"])
#     # print("Precision Val data  {:.4f}%".format(precision*100))
#     # print("Recall Val data {:.4f}%".format(recall*100))

#     # fpr, tpr, thresholds = roc_curve(data_write['BAD_FLAG'], data_write["Predicted_probability"])
#     # roc_auc = auc(fpr, tpr)
#     # GINI_Val = (2 * roc_auc) - 1
#     # print(f'Gini on Val data with XGBoost: {GINI_Val}')

#     # write back to snowflake:
#     truncate_table("dod_results", "cm_ac_vanilla_v1".lower(), cur)
#     # write temp DoD table:
#     write_to_snowflake(data_write, "dod_results", "cm_ac_vanilla_v1".lower())

#     split_data = get_raw_data(start_date, end_date, cur, gd.get_split_data)
#     print(f"*********** split data for DS mode shape : {split_data.shape} ***********")

#     exclude = [
#         "RISK_FLAG",
#         "ONE_DPD_TELE",
#         "FINAL_RISK_FLAG",
#         "RISK_RANK",
#         "PERCENTILE",
#         "ROW_NUM",
#         "MODEL",
#     ]

#     split_data_1 = split_data.drop(columns=exclude)
#     data_write.rename(columns={"FULL_DATE": "UPDATE_DT"}, inplace=True)
#     # converting all columns to UPPER CASE
#     data_write.columns = data_write.columns.str.upper()

#     split_data_pred = split_data_1.merge(
#         data_write[
#             [
#                 "LOAN_ID",
#                 "UPDATE_DT",
#                 "PREDICTED_PROBABILITY",
#                 "PREDICTED_BINARY_VERDICT",
#             ]
#         ],
#         how="inner",
#         on=["LOAN_ID", "UPDATE_DT"],
#     )

#     split_data_pred_all = copy.copy(split_data_pred)
#     split_data_pred_all["GROUP_NAME"] = "Treatment"

#     split_data_pred_all["RISK_FLAG"] = np.where(
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 0, "Non-Risky", "Risky"
#     )
#     split_data_pred_all_write = split_data_pred_all[
#         [
#             "UPDATE_DT",
#             "LOAN_ID",
#             "TODAYS_EDI",
#             "PREDICTED_PROBABILITY",
#             "RISK_FLAG",
#             "GROUP_NAME",
#         ]
#     ]

#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred_all_write.shape} ***********"
#     )

#     split_data_pred_all_write.rename(
#         columns={"PREDICTED_PROBABILITY": "PROBABILITY_OF_RISKY"}, inplace=True
#     )

#     truncate_table(
#         "dod_results_ab_test_full_write_monitoring", "cm_ac_vanilla_v1".lower(), cur
#     )
#     write_to_snowflake(
#         split_data_pred_all_write,
#         "dod_results_ab_test_full_write_monitoring",
#         "cm_ac_vanilla_v1".lower(),
#     )

#     split_data_pred_risky = split_data_pred.loc[
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 1, :
#     ]
#     split_data_pred_risky_share = split_data_pred_risky.drop(
#         columns=["PREDICTED_BINARY_VERDICT", "PREDICTED_PROBABILITY"]
#     )

#     split_data_pred_risky_share["DB_UPDATED_AT"] = split_data_pred_risky_share[
#         "DB_UPDATED_AT"
#     ].astype("str")

#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred.shape} ***********"
#     )
#     print(
#         f"*********** predicted split data for DS mode  : {split_data_pred.head()} ***********"
#     )
#     print(
#         f"*********** Risky predicted split data for DS mode shape  : {split_data_pred_risky_share.shape} ***********"
#     )

#     split_data_pred_risky_share["RUN_TIMESTAMP"] = pd.to_datetime(datetime.now())
#     split_data_pred_risky_share["RUN_TIMESTAMP"] = split_data_pred_risky_share[
#         "RUN_TIMESTAMP"
#     ].astype("str")

#     truncate_table("dod_results_ab_test", "cm_ac_vanilla_v1".lower(), cur)
#     write_to_snowflake(
#         split_data_pred_risky_share, "dod_results_ab_test", "cm_ac_vanilla_v1".lower()
#     )

#     # writing to common combination table
#     trigger_cmd(gd.write_to_combination_table, conn)

#     # writing to master table for monitoring
#     trigger_cmd(gd.write_split_full_data, conn)

#     # merging prediction to final table
#     trigger_cmd(gd.merge_predicted_data, conn)

#     # merge back to master predicted result table for 65% predicted risky of 50% total
#     trigger_cmd(gd.merge_predicted_split_data, conn)

#     cur.close()
#     conn.close()


# def model_run():
#     start_date = datetime.now().strftime("%Y-%m-%d")
#     end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
#     # get data
#     data = get_raw_data(start_date, end_date, cur, gd.query)

#     # pick only selected features utilized in model
#     feature_list = pd.read_csv(
#         read_file(uf.s3_bucket, uf.model_path + "XGBoost_feature_list_100_prev.csv")
#     )["variables"]
#     # select only cols used in model
#     data_filter = data[list(feature_list)]
#     # remove_inconsistent_values(data)
#     data_filter = remove_inconsistent_values(data_filter, feature_list)

#     data_filter = gd.missing_ind_convert_num(data_filter)
#     # unpickle cm_ac_vanilla_v1 model
#     model = pickle.loads(
#         uf.s3.Bucket(uf.s3_bucket)
#         .Object(f"{uf.model_path}Vanilla_model_v1.pkl")
#         .get()["Body"]
#         .read()
#     )

#     # run predictions and store the result
#     data["Predicted_probability"] = model.predict_proba(data_filter)[:, 1]

#     # finding threshold value for 35% prediction coverage
#     threshold_value = uf.find_threshold(data, uf.cut_off, "DS")

#     # converting threshold cut to binary
#     data["Predicted_binary_verdict"] = np.where(
#         data["Predicted_probability"] >= threshold_value, 1, 0
#     )

#     # store only relevant info:
#     data_write = data[
#         list(feature_list)
#         + [
#             "LOAN_ID",
#             "USER_ID",
#             "FULL_DATE",
#             "Predicted_probability",
#             "Predicted_binary_verdict",
#             "AUTO_CURE_FLAG",
#         ]
#     ]

#     # precision = precision_score(data_write['AUTO_CURE_FLAG'], data_write["Predicted_binary_verdict"])
#     # # Calculate recall
#     # recall = recall_score(data_write['AUTO_CURE_FLAG'], data_write["Predicted_binary_verdict"])
#     # print("Precision Val data  {:.4f}%".format(precision*100))
#     # print("Recall Val data {:.4f}%".format(recall*100))

#     # fpr, tpr, thresholds = roc_curve(data_write['AUTO_CURE_FLAG'], data_write["Predicted_probability"])
#     # roc_auc = auc(fpr, tpr)
#     # GINI_Val = (2 * roc_auc) - 1
#     # print(f'Gini on Val data with XGBoost: {GINI_Val}')

#     # write back to snowflake:
#     truncate_table("dod_results", "cm_ac_vanilla_v1".lower(), cur)
#     # write temp DoD table:
#     write_to_snowflake(data_write, "dod_results", "cm_ac_vanilla_v1".lower())

#     get_split_data = """
#         select 
#             *
#         from 
#             ANALYTICS.KB_ANALYTICS.DAILY_SPLIT_AB_TEST_COLLECTIONS_MODEL_DPD1
#         where
#             1=1
#             and MODEL = 'DATA_SCIENCE'
#             and UPDATE_DT = CURRENT_DATE()
#     """

#     split_data = get_raw_data(start_date, end_date, cur, get_split_data)
#     print(f"*********** split data for DS mode shape : {split_data.shape} ***********")

#     exclude = [
#         "RISK_FLAG",
#         "ONE_DPD_TELE",
#         "FINAL_RISK_FLAG",
#         "RISK_RANK",
#         "PERCENTILE",
#         "ROW_NUM",
#         "MODEL",
#     ]

#     split_data_1 = split_data.drop(columns=exclude)
#     data_write.rename(columns={"FULL_DATE": "UPDATE_DT"}, inplace=True)
#     # converting all columns to UPPER CASE
#     data_write.columns = data_write.columns.str.upper()

#     split_data_pred = split_data_1.merge(
#         data_write[
#             [
#                 "LOAN_ID",
#                 "UPDATE_DT",
#                 "PREDICTED_PROBABILITY",
#                 "PREDICTED_BINARY_VERDICT",
#             ]
#         ],
#         how="inner",
#         on=["LOAN_ID", "UPDATE_DT"],
#     )

#     split_data_pred_all = copy.copy(split_data_pred)
#     split_data_pred_all["GROUP_NAME"] = "Treatment"
#     split_data_pred_all["RISK_FLAG"] = np.where(
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 0, "Risky", "Non-Risky"
#     )
#     split_data_pred_all_write = split_data_pred_all[
#         [
#             "UPDATE_DT",
#             "LOAN_ID",
#             "TODAYS_EDI",
#             "PREDICTED_PROBABILITY",
#             "RISK_FLAG",
#             "GROUP_NAME",
#         ]
#     ]

#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred_all_write.shape} ***********"
#     )

#     split_data_pred_all_write.rename(
#         columns={"PREDICTED_PROBABILITY": "PROBABILITY_OF_RISKY"}, inplace=True
#     )

#     truncate_table(
#         "dod_results_ab_test_full_write_monitoring", "cm_ac_vanilla_v1".lower(), cur
#     )
#     write_to_snowflake(
#         split_data_pred_all_write,
#         "dod_results_ab_test_full_write_monitoring",
#         "cm_ac_vanilla_v1".lower(),
#     )

#     split_data_pred_risky = split_data_pred.loc[
#         split_data_pred["PREDICTED_BINARY_VERDICT"] == 0, :
#     ]
#     split_data_pred_risky_share = split_data_pred_risky.drop(
#         columns=["PREDICTED_BINARY_VERDICT", "PREDICTED_PROBABILITY"]
#     )

#     split_data_pred_risky_share["DB_UPDATED_AT"] = split_data_pred_risky_share[
#         "DB_UPDATED_AT"
#     ].astype("str")

#     print(
#         f"*********** predicted split data for DS mode shape  : {split_data_pred.shape} ***********"
#     )
#     print(
#         f"*********** predicted split data for DS mode  : {split_data_pred.head()} ***********"
#     )
#     print(
#         f"*********** Risky predicted split data for DS mode shape  : {split_data_pred_risky_share.shape} ***********"
#     )

#     split_data_pred_risky_share["RUN_TIMESTAMP"] = pd.to_datetime(datetime.now())
#     split_data_pred_risky_share["RUN_TIMESTAMP"] = split_data_pred_risky_share[
#         "RUN_TIMESTAMP"
#     ].astype("str")

#     truncate_table("dod_results_ab_test", "cm_ac_vanilla_v1".lower(), cur)
#     write_to_snowflake(
#         split_data_pred_risky_share, "dod_results_ab_test", "cm_ac_vanilla_v1".lower()
#     )

#     # writing to common combination table
#     trigger_cmd(gd.write_to_combination_table, conn)

#     # writing to master table for monitoring
#     trigger_cmd(gd.write_split_full_data, conn)

#     # merging prediction to final table
#     trigger_cmd(gd.merge_predicted_data, conn)

#     # merge back to master predicted result table for 65% predicted risky of 50% total
#     trigger_cmd(gd.merge_predicted_split_data, conn)

#     cur.close()
#     conn.close()


# model_run()


# # scrap run

# start_date =  datetime.now().strftime("%Y-%m-%d")
# end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

# dod_results = """
#     select
#         *
#     from
#         ANALYTICS.KB_ANALYTICS.DOD_RESULTS_CM_AC_VANILLA_V1_COLLECTIONSMODEL_V1;
# """


# get_split_data = """
#     select
#         *
#     from
#         ANALYTICS.KB_ANALYTICS.DAILY_SPLIT_AB_TEST_COLLECTIONS_MODEL_DPD1
#     where
#         1=1
#         and MODEL = 'DATA_SCIENCE'
#         and UPDATE_DT = CURRENT_DATE()
# """

# data_write = get_raw_data(start_date,end_date,cur,dod_results)

# data_write.shape

# data_write.PREDICTED_BINARY_VERDICT.value_counts()

# split_data.UPDATE_DT.value_counts()

# split_data = get_raw_data(start_date,end_date,cur,get_split_data)

# print(f"*********** split data for DS mode shape : {split_data.shape} ***********")

# exclude = ['RISK_FLAG','ONE_DPD_TELE','FINAL_RISK_FLAG','RISK_RANK','PERCENTILE','ROW_NUM','MODEL']

# split_data_1 = split_data.drop(columns=exclude)
# data_write.rename(columns={"FULL_DATE":"UPDATE_DT"},inplace=True)

# # convert data_write_cols to all caps
# data_write.columns = data_write.columns.str.upper()

# split_data_1['UPDATE_DT'] = split_data['UPDATE_DT'].astype('str')
# data_write['UPDATE_DT'] = data_write['UPDATE_DT'].astype('str')

# common = set(data_write['LOAN_ID']).intersection(split_data_1['LOAN_ID'])
# len(common)

# split_data_pred = split_data_1.merge(data_write[['LOAN_ID','UPDATE_DT','PREDICTED_PROBABILITY','PREDICTED_BINARY_VERDICT']],how = 'inner',on =['LOAN_ID','UPDATE_DT'])

# split_data_pred.shape
# import copy 
# split_data_pred_all = copy.copy(split_data_pred)
# split_data_pred_all['GROUP_NAME'] = 'Treatment'
# split_data_pred_all['RISK_FLAG'] = np.where(split_data_pred['PREDICTED_BINARY_VERDICT']==0,"Risky","Non-Risky")
# split_data_pred_all_write = split_data_pred_all[['UPDATE_DT','LOAN_ID','TODAYS_EDI','PREDICTED_PROBABILITY','RISK_FLAG','GROUP_NAME']]

# split_data_pred_all_write.shape
# split_data_pred_all_write.rename(columns={"PREDICTED_PROBABILITY":"PROBABILITY_OF_RISKY"},inplace=True)
# split_data_pred_all_write['RISK_FLAG'].value_counts()

# truncate_table("dod_results_ab_test_full_write_monitoring", "cm_ac_vanilla_v1".lower(), cur)
# write_to_snowflake(
#         split_data_pred_all_write, "dod_results_ab_test_full_write_monitoring", "cm_ac_vanilla_v1".lower()
#     )

# write_split_full_data = """
#     INSERT INTO ANALYTICS.ADHOC.MONITORING_DAILY_TELECALLING_DATA_COMBINE_COLLECTIONS_MODEL
#     SELECT *
#     FROM ANALYTICS.KB_ANALYTICS.DOD_RESULTS_AB_TEST_FULL_WRITE_MONITORING_CM_AC_VANILLA_V1_COLLECTIONSMODEL_V1;
# """
# trigger_cmd(write_split_full_data, conn)

# split_data_pred.columns




