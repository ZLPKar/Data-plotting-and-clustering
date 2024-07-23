import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def group_less_frequent(df, column, threshold=100):
    freq_items = df[column].value_counts().index[df[column].value_counts() >= threshold]
    df[column] = df[column].where(df[column].isin(freq_items), other='Other')

def age_readmitted_graph(df_diabetic):
    age_groups = pd.cut(df_diabetic['age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'])
    age_groups = pd.Categorical(age_groups, categories=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'], ordered=True)
    age_readmission = df_diabetic.groupby(age_groups)["readmitted"].mean()

    plt.figure(figsize=(10, 6))
    plt.bar(age_readmission.index, age_readmission.values)
    plt.title("Age vs. Readmission")
    plt.xlabel("Age")
    plt.ylabel("Mean Readmission Rate")
    plt.xticks(rotation=45)
    plt.show()


def race_readmitted_graph(df_diabetic):
    group_less_frequent(df_diabetic, "race")
    race_readmission = df_diabetic.groupby("race")["readmitted"].mean()
    plt.figure(figsize=(10, 6))
    plt.bar(race_readmission.index, race_readmission.values)
    plt.title("Race vs. Readmission")
    plt.xlabel("Race")
    plt.ylabel("Mean Readmission Rate")
    plt.show()

def gender_readmitted_graph(df_diabetic):
    gender_readmission = df_diabetic.groupby("gender")["readmitted"].mean()
    plt.figure(figsize=(10, 6))
    plt.bar(gender_readmission.index, gender_readmission.values)
    plt.title("Gender vs. Readmission")
    plt.xlabel("Gender")
    plt.ylabel("Mean Readmission Rate")
    plt.show()

def diag1_readmitted_graph(df_diabetic):
    group_less_frequent(df_diabetic, "Description", threshold=100)
    diag_1_readmission = df_diabetic.groupby("Description")["readmitted"].mean()
    plt.figure(figsize=(10, 6))
    plt.bar(diag_1_readmission.index, diag_1_readmission.values)
    plt.title("Diagnosis Type (diag_1) vs. Readmission")
    plt.xlabel("Diagnosis Type")
    plt.ylabel("Mean Readmission Rate")
    plt.xticks(rotation=90)
    plt.show()

def diag2_readmitted_graph(df_diabetic):
    group_less_frequent(df_diabetic, "Description_diag_2", threshold=100)
    diag_2_readmission = df_diabetic.groupby("Description_diag_2")["readmitted"].mean()
    plt.figure(figsize=(10, 6))
    plt.bar(diag_2_readmission.index, diag_2_readmission.values)
    plt.title("Diagnosis Type (diag_2) vs. Readmission")
    plt.xlabel("Diagnosis Type")
    plt.ylabel("Mean Readmission Rate")
    plt.xticks(rotation=90)
    plt.show()

def diag3_readmitted_graph(df_diabetic):
    group_less_frequent(df_diabetic, "Description_diag_3", threshold=100)
    diag_3_readmission = df_diabetic.groupby("Description_diag_3")["readmitted"].mean()
    plt.figure(figsize=(10, 6))
    plt.bar(diag_3_readmission.index, diag_3_readmission.values)
    plt.title("Diagnosis Type (diag_3) vs. Readmission")
    plt.xlabel("Diagnosis Type")
    plt.ylabel("Mean Readmission Rate")
    plt.xticks(rotation=90)
    plt.show()

def main(df_diabetic):
    age_readmitted_graph(df_diabetic)
    race_readmitted_graph(df_diabetic)
    gender_readmitted_graph(df_diabetic)
    diag1_readmitted_graph(df_diabetic)
    diag2_readmitted_graph(df_diabetic)
    diag3_readmitted_graph(df_diabetic)
