## Data processing
import pandas as pd
import numpy as np
import collections_model_get_data_v1 as gd
from collections_model_utility_functions_v1 import get_connector, get_raw_data, truncate_table, write_to_snowflake, read_file
import collections_model_utility_functions_v1 as uf
# Model and performance evaluation
from sklearn.metrics import roc_curve, auc, precision_score, recall_score
## unpickling the model
import pickle



conn = get_connector()
cur = conn.cursor()

def identify_column_type(series):
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'
    elif pd.api.types.is_categorical_dtype(series):
        return 'categorical'
    else:
        return 'other'

def remove_inconsistent_values(df,feature_list):
    for column in feature_list:
        column_type = identify_column_type(df[column])
        if column_type == 'numeric':
            df[column] = pd.to_numeric(df[column], errors='coerce')
            # df = df.dropna(subset=[column])
        elif column_type == 'categorical':
            # df[column] = df[column].astype('category')
            df[column] = pd.to_numeric(df[column], errors='coerce')
        else:
            # Handle other column types if needed
            df[column] = pd.to_numeric(df[column], errors='coerce')
            # pass
    return df

def model_run():
    start_date = '2023-06-02'
    end_date = '2023-06-03'
    # get data
    data= get_raw_data(start_date,end_date,cur,gd.query)

    # pick only selected features utilized in model
    feature_list=pd.read_csv(
    read_file(uf.s3_bucket, uf.model_path + "XGBoost_feature_list_100_prev.csv")
    )['variables']
    # select only cols used in model
    data_filter = data[list(feature_list)]
    # remove_inconsistent_values(data)
    data_filter = remove_inconsistent_values(data_filter,feature_list)

    data_filter = gd.missing_ind_convert_num(data_filter)
    # unpickle cm_ac_vanilla_v1 model 
    model=pickle.loads(
                uf.s3.Bucket(uf.s3_bucket).Object(f"{uf.model_path}Vanilla_model_v1.pkl").get()["Body"].read()
            ) 

    # run predictions and store the result 
    data['Pred_PD']=model.predict_proba(data_filter)[:,1]

    # finding threshold value for 35% prediction coverage
    threshold_value = uf.find_threshold(data,uf.cut_off)

    # converting threshold cut to binary
    data['Pred_PD_binary_tc_thresh'] = np.where(data['Pred_PD']>threshold_value,1,0)


    # store only relevant info:
    data_write = data[list(feature_list)+['LOAN_ID','USER_ID','FULL_DATE','Pred_PD','Pred_PD_binary_tc_thresh','AUTO_CURE_FLAG']]

    precision = precision_score(data_write['AUTO_CURE_FLAG'], data_write["Pred_PD_binary_tc_thresh"])
    # Calculate recall
    recall = recall_score(data_write['AUTO_CURE_FLAG'], data_write["Pred_PD_binary_tc_thresh"])
    print("Precision Val data  {:.4f}%".format(precision*100))
    print("Recall Val data {:.4f}%".format(recall*100))

    fpr, tpr, thresholds = roc_curve(data_write['AUTO_CURE_FLAG'], data_write["Pred_PD"])
    roc_auc = auc(fpr, tpr)
    GINI_Val = (2 * roc_auc) - 1
    print(f'Gini on Val data with XGBoost: {GINI_Val}')

    # write back to snowflake:
    truncate_table("dod_results", "cm_ac_vanilla_v1".lower(),cur)

    # write temp DoD table:
    write_to_snowflake(data_write, "dod_results", "cm_ac_vanilla_v1".lower())

    # merge back to master result table
    # merge()
    cur.close() 
    conn.close()

model_run()

