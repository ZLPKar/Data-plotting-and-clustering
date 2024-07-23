import pandas as pd
import numpy as np

numerical_features = [
    "encounter_id", "patient_nbr", "time_in_hospital", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses"
]

categorical_features = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "payer_code", "medical_specialty", "diag_1", "diag_2",
    "diag_3", "max_glu_serum", "A1Cresult", "change", "diabetesMed",
    "medications"
]

def drop_missing_value_columns(df):
    threshold_missing = len(df) * 0.5
    df.dropna(thresh=threshold_missing, axis=1, inplace=True)

def drop_same_value_columns(df):
    threshold_same_values = len(df) * 0.95
    columns_to_drop = []
    for column in df.columns:
        if df[column].nunique() == 1 or (df[column].value_counts().max() > threshold_same_values):
            columns_to_drop.append(column)
    df.drop(columns=columns_to_drop, inplace=True)

def transform_age_range(age_range):
    lower, upper = map(int, age_range.strip("[]()").split("-"))
    middle_value = (lower + upper) // 2
    return middle_value

def remove_outliers(df, numerical_features):
    means = df[numerical_features].mean()
    stds = df[numerical_features].std()

    for feature in numerical_features:
        lower_bound = means[feature] - 3 * stds[feature]
        upper_bound = means[feature] + 3 * stds[feature]
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    
def main(df, df_icd_codes):
    print("dataframe shape:", df.shape)

    df.replace("?", np.nan, inplace=True)

    # Drop all columns that have more than 50% of missing values.
    drop_missing_value_columns(df)

    # You can also drop columns for which over 95% of their values are the same.
    drop_same_value_columns(df)

    # Ages are given in a 10 years range (i.e. [10-20)).
    # Transform these ages to be the middle value in each given range
    df["age"] = df["age"].apply(transform_age_range)

    # Replace possible missing values in the columns diag_1, diag_2, and diag_3 by the number 0.
    df[["diag_1", "diag_2", "diag_3"]] = df[["diag_1", "diag_2", "diag_3"]].fillna(0)

    # Drop all rows with missing values.
    df.dropna(inplace=True)

    # Identify outliers in the numerical columns and remove them.
    remove_outliers(df, numerical_features)

    # Remove duplicates in the column patient_nbr and show the shape of the resulting dataframe
    df.drop_duplicates(subset=['patient_nbr'], keep='first', inplace=True)
    print("dataframe shape:", df.shape)

    # Transform readmitted values to binary
    df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x in ["<30", ">30"] else 0)

    # Merge df_diabetic with df_icd_codes based on diag_1
    df = df.merge(df_icd_codes, left_on="diag_1", right_on="ICD_Code", how="left", suffixes=("", "_diag_1"))

    # Merge df_diabetic with df_icd_codes based on diag_2,
    df = df.merge(df_icd_codes, left_on="diag_2", right_on="ICD_Code", how="left", suffixes=("", "_diag_2"))

    # Merge df_diabetic with df_icd_codes based on diag_3
    df = df.merge(df_icd_codes, left_on="diag_3", right_on="ICD_Code", how="left", suffixes=("", "_diag_3"))

    # Replace readmitted column values for 100% consistency
    df["readmitted"] = df["readmitted"].replace({"NO": 0, "<30": 1, ">30": 1})

    return df