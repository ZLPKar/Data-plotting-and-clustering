import pandas as pd
import data_cleaning
import data_exploration
import first_model
import improved_model

if __name__ == "__main__":
    show_visualisations = True
    model_picker = 2 # 0: first model, 1: improved model, 2: both models, anything else: no models

    df_icd_codes = pd.read_csv("files/icd_codes.csv", header=0)
    df_diabetic = pd.read_csv("files/diabetic_data.csv", header=0)
    
    df_diabetic = data_cleaning.main(df_diabetic, df_icd_codes)

    if show_visualisations:
        data_exploration.main(df_diabetic)

    if model_picker == 0:
        print("\nFirst Model:")
        first_model.main(df_diabetic)
    elif model_picker == 1:
        print("\nImproved Model:")
        improved_model.main(df_diabetic)
    elif model_picker == 2:
        print("\nFirst Model:")
        first_model.main(df_diabetic)
        print("\nImproved Model:")
        improved_model.main(df_diabetic)
