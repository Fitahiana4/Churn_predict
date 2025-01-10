import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# ALTAIR
import altair as alt

# STREMLIT APP
import streamlit as st

import pickle

import os


# dataset exapmle
data_exampler = pd.read_csv('Datachurn_inf.csv')
data_exampler


columns_to_encoded = ['City','Churn']

all_columns = ['City',
 'FinishedTrips',
 'Amount_per_trip',
 'TripDistance_per_trip',
 'prop_voiture',
 'prop_tricycle',
 'prop_moto',
 'completion_rate',
 'user_canc_rate',
 'admin_canc_rate',
 'champ_canc_rate',
 'supply_rate',
 'Request',
 'Churn']

def make_encoding_labelencoder(df, columns):
    label_encoder = LabelEncoder()
    for col in columns:
        df[col] = label_encoder.fit_transform(df[col])
    return df

def make_data_encoded(data, is_one_line=True):
    if is_one_line:
        data = pd.DataFrame(data, index=[0])
    data["Churn"] = data["Churn"].map({'No Churn': 0, 'Churn': 1})

    return data


filtered_columns = ['City',
 'FinishedTrips',
 'Amount_per_trip',
 'TripDistance_per_trip',
 'prop_voiture',
 'prop_tricycle',
 'prop_moto',
 'completion_rate',
 'user_canc_rate',
 'admin_canc_rate',
 'champ_canc_rate',
 'supply_rate',
 'Request']


def prediction_multi_lines(df_churn):
    try:
        prediction_results = churn_model.predict(df_churn[filtered_columns])
        df_pred = pd.DataFrame({'Prediction': np.array(prediction_results).flatten()})
        # Create a new column 'Classification' based on the 'Prediction' column
        df_pred['Classification'] = df_pred['Prediction'].map({1: 'Churner', 0: 'Loyal'})
        # Count the occurrences of each classification
        classification_counts = df_pred['Classification'].value_counts()
        # classification_counts = df_pred['Prediction'].value_counts()
        classification_counts = pd.DataFrame(classification_counts)

        # ------------------------
        # Result
        col1, col2 = st.columns(2)
        with col1:
            df_pred
            # st.write(classification_counts["count"])
        with col2:
            st.bar_chart(classification_counts)

    except ValueError as e:
        st.error(f"Prediction error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def prepare_data_and_predict(df_uploaded):
    # Virify whether all columns in column_list are present in the dataset
    missing_columns = [col for col in all_columns if col not in df_uploaded.columns]
    if missing_columns:
            st.error(f"The following columns are missing in the dataset: {missing_columns}")
    else:
        st.write(df_uploaded.head(3))
        # shape of date
        st.write(f"**Data size**: {df_uploaded.shape[0]}")
        st.header("Result of Churn prediction", divider=True)
        # accept the prediction whether the dataset'size is more than 1
        if df_uploaded.shape[0] > 0:
            pass

        else:
                try:
                    # manual encoding fata
                    df_churn = make_data_encoded(data=df_churn, is_one_line=False)
                    # # making standard scaler
                    # df_churn = manual_standardscaler(df=df_churn, columns=columns_params)
                    # making prediction
                    prediction_multi_lines(df_churn)
                except:
                    st.info("###### Can't support the this data.")

def get_file_extension(file_path):
    _, extension = os.path.splitext(file_path.name)
    return f"{extension}"

#pip install streamlit

model_path = './churn_model.pkl'

churn_model = pickle.load(open(model_path, 'rb'))

# ---------------------------------
#   A P P L I C A T I O N
# ---------------------------------
st.title("🤖 Machine Learning")

with st.expander("CHURN PREDICTION - BY UPLOADIN FILE"):
    st.write("***Your data must content some columns(11) like this :***")
    st.dataframe(data_exampler)
    df_uploaded = st.file_uploader(label="Upload the dataset here")
    if df_uploaded:
        if get_file_extension(df_uploaded) == ".csv":
            df_uploaded = pd.read_csv(df_uploaded)
            prepare_data_and_predict(df_uploaded)
        elif get_file_extension(df_uploaded) == ".xlsx":
            df_uploaded = pd.read_excel(df_uploaded)
            prepare_data_and_predict(df_uploaded)

        else:
            st.error("#### Make sure you had uploaded csv or excel file")















