#20250811添加对字符串标签的处理

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os


def fen_process(input_file, ID_num, custom_bins=None):

    df = pd.read_csv(input_file, header=0)
    label_column = df.columns[-1]

    if custom_bins is None:
        custom_bins = {}

    for column in df.columns[:-1]:
        data = df[column]

        if data.dtype == 'object':
            continue

        elif np.issubdtype(data.dtype, np.number):

            bins_num = custom_bins.get(column, ID_num)

            if len(data.unique()) > bins_num:
                try:
                    df[column] = pd.qcut(data, q=bins_num, duplicates='drop')
                except Exception as e:
                    print(f"[WARNING] Feature {column} binning failed. Reason: {e}")

        else:
            print(f"Column '{column}' has unsupported data type: {data.dtype}")

    return df



def id_and_save_csv(df, csv_output_file):
    value_dict = {}
    current_value = 0

    label_column = df.columns[-1]
    
    # Check if the label column is a string and convert to IDs if necessary
    if df[label_column].dtype == 'object':
        print(f"[INFO] Converting label column '{label_column}' to ID.")
        label_encoder = LabelEncoder()
        df[label_column] = label_encoder.fit_transform(df[label_column])

    for col in df.columns:
        if col == label_column:
            continue

        else:
            value_mapping = {}
            unique_values = df[col].unique()
            for i, val in enumerate(unique_values):
                value_mapping[val] = current_value + i

            current_value += len(unique_values)
            df[col] = df[col].map(value_mapping)
            value_dict[col] = value_mapping

    num_features = df.shape[1] - 1
    df.columns = [f'f{i+1}' for i in range(num_features)] + ['label']
    csv_output_file = os.path.join(csv_output_file, 'data_ID.csv')
    df.to_csv(csv_output_file, index=False)
    print(f"Processing complete. Output ID file: {csv_output_file}")

    return df, value_dict

def save_mapping_as_csv(value_dict, raw_df, dict_csv_file):

    rows = []
    total = len(value_dict)
    for i, (col, mapping) in enumerate(value_dict.items()):

        feature_name = f"f{i+1}_{col}"
        series = raw_df[col]

        for orig, mapped in mapping.items():
            count = int(series.value_counts().get(orig, 0))
            rows.append({
                'feature':        feature_name,
                'original_value': orig,
                'mapped_value':   mapped,
                'count':          count
            })
        rows.append({'feature':'', 'original_value':'', 'mapped_value':'', 'count':''})

    df_map = pd.DataFrame(rows, columns=['feature', 'original_value', 'mapped_value', 'count'])
    dict_csv_file = os.path.join(dict_csv_file, 'data_mapping.csv')
    df_map.to_csv(dict_csv_file, index=False, encoding='utf-8-sig')
    print(f"Processing complete. Output ID mapping CSV file: {dict_csv_file}")


def save_summary_as_csv(original_df, processed_df, value_dict, summary_output_file):
    summary = []
    label_col = original_df.columns[-1]
    
    for idx, col in enumerate(original_df.columns):

        if col == label_col:

            if original_df[col].dtype == 'object':
                col_type = 1
                min_val = '-'
                max_val = '-'
            else:
                col_type = 0
                min_val = original_df[col].min()
                max_val = original_df[col].max()
            unique_count = original_df[col].nunique()
            id_size = len(processed_df[f'label'].unique())
            
            summary.append([
                f"label",
                col_type,
                min_val,
                max_val,
                unique_count,
                id_size
            ])

        else:
            if original_df[col].dtype == 'object':
                col_type = 1
                min_val = '-'
                max_val = '-'
            else:
                col_type = 0
                min_val = original_df[col].min()
                max_val = original_df[col].max()
            unique_count = original_df[col].nunique()
            id_size = len(processed_df[f'f{idx+1}'].unique())
            
            summary.append([
                f"Feature_{idx + 1}",
                col_type,
                min_val,
                max_val,
                unique_count,
                id_size
            ])

    summary_df = pd.DataFrame(summary, columns=[' ','str_or_number', 'Min', 'Max', 'Unique Count', 'id_size'])
    summary_output_file = os.path.join(summary_output_file, 'data_summary.csv')
    summary_df.to_csv(summary_output_file, index=False)
    print(f"Processing complete. Output data summary file: {summary_output_file}")



if __name__ == "__main__":
    
    input_file =  '1153.csv'     # Original input file
    output_csv =  './ID_07_02'
    csv_mapping_file = output_csv
    summary_file = output_csv

    global_bins  = 3
    # custom_bins = {"Age": 10}
    custom_bins = {}
    
    os.makedirs(output_csv, exist_ok=True)
    os.makedirs(csv_mapping_file, exist_ok=True)
    os.makedirs(summary_file, exist_ok=True)

    df_raw = pd.read_csv(input_file) 
    df_fen = fen_process(input_file, global_bins, custom_bins)
    df_fen_copy = df_fen.copy(deep=True)
    df_id, val_dict = id_and_save_csv(df_fen, output_csv)
    save_mapping_as_csv(val_dict, df_fen_copy, csv_mapping_file)
    save_summary_as_csv(df_raw, df_id, val_dict, summary_file)