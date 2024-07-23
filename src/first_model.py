import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

def preprocess_data(df):
    subset_columns = ['num_medications', 'number_outpatient', 'number_emergency', 
                      'time_in_hospital', 'number_inpatient', 'encounter_id', 
                      'age', 'num_lab_procedures', 'number_diagnoses', 
                      'num_procedures', 'readmitted']
    df = df[subset_columns]

    df['readmitted'].apply(lambda x: 1 if x in ['<30', '>30'] else 0)

    df = df.dropna()

    df['age'] = df['age'] + 5

    df.drop_duplicates(subset=['encounter_id'], keep='first', inplace=True)

    return df

def split_data(df):
    X = df.drop(columns=['readmitted'])
    y = df['readmitted']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Mean cross-validation score:", cv_scores.mean())

def calculate_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

def evaluate_model2(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


def main(df):
    df = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Build model
    model = build_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_train, y_train)
    evaluate_model2(model, X_test, y_test)
    calculate_confusion_matrix(model, X_test, y_test)

if __name__ == "__main__":
    main()
