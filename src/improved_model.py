import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

def balance_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def preprocess_data(df, subset_columns):
    df = df[subset_columns]
    df = pd.get_dummies(df, columns=['race', 'gender', 'change'])
    subset_columns = list(df.columns)
    return df, subset_columns

def split_data(df):
    X = df.drop(columns=['readmitted'])
    y = df['readmitted']
    return train_test_split(X, y, test_size=0.15, random_state=42)

def build_random_forest_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt'] 
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)
    best_model = grid_search.best_estimator_
    return best_model

def evaluate_model(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Mean cross-validation score:", cv_scores.mean())

def evaluate_model2(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

def cluster_data(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X)
    return clusters

def visualise_clusters(X, clusters, column1, column2):
    plt.scatter(X[column1], X[column2], c=clusters, cmap='viridis')
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.title('K-Means Clustering')
    plt.colorbar(label='Cluster')
    plt.show()

def visualize_cluster_distribution(X, clusters, feature_names):
    num_clusters = len(np.unique(clusters))
    num_features = X.shape[1]

    print(num_features, len(feature_names))

    for feature_index in range(num_features):
        plt.figure(figsize=(10, 6))
        for cluster_label in range(num_clusters):
            plt.hist(X.iloc[clusters == cluster_label, feature_index], bins=20, alpha=0.5, label=f'Cluster {cluster_label}')
        plt.hist(X.iloc[:, feature_index], bins=20, alpha=0.5, label='Overall')
        plt.xlabel(feature_names[feature_index])
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {feature_names[feature_index]} in Clusters')
        plt.legend()
        plt.show()


def main(df):
    feature_names = ['num_medications', 'number_outpatient', 'number_emergency', 
                      'time_in_hospital', 'number_inpatient', 'encounter_id', 
                      'age', 'num_lab_procedures', 'number_diagnoses', 'change', 
                      'num_procedures', 'readmitted', 'race', 'gender']

    df, feature_names = preprocess_data(df, feature_names)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train_balanced, y_train_balanced = balance_data(X_train, y_train)

    rf_model = build_random_forest_model(X_train_balanced, y_train_balanced)
    evaluate_model(rf_model, X_train_balanced, y_train_balanced)
    evaluate_model2(rf_model, X_test, y_test)

    clusters = cluster_data(X_train_balanced, n_clusters=5)
    df['clust'] = pd.Series(clusters)
    visualise_clusters(X_train_balanced, clusters, 'num_medications', 'time_in_hospital')
    visualise_clusters(X_train_balanced, clusters, 'num_lab_procedures', 'number_diagnoses')
    visualise_clusters(X_train_balanced, clusters, 'number_outpatient', 'number_inpatient')
    visualize_cluster_distribution(X_train_balanced, clusters, feature_names)
