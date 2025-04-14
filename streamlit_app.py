###############################################
#           streamlit_app.py
###############################################
import streamlit as st
import pandas as pd
import numpy as np
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# ------------------------------
# 1) Parsing Utility
# ------------------------------
def parse_string_list(str_list):
    """Safely evaluate a string that looks like a Python list (e.g., "['fever','stress']")"""
    try:
        return ast.literal_eval(str_list)
    except:
        return []

# ------------------------------
# 2) Caching Data Load
# ------------------------------
@st.cache_data
def load_data(csv_path="disease_features.csv"):
    """
    Loads the disease dataset from CSV. 
    Parses textual columns (Risk Factors, Symptoms, Signs) and 
    creates a combined TF-IDF feature matrix (3 columns merged).
    Trains a KNN model on the entire dataset (for demonstration).
    Returns:
      - disease_df: the full DataFrame
      - tfidf_matrix: the combined TF-IDF feature matrix (sparse)
      - label_encoder: fitted LabelEncoder for disease labels
      - vectorizers: dictionary of TfidfVectorizer objects {col_name: vectorizer}
      - knn_model: the default KNN model (k=3, metric='euclidean') that we can re-fit with user choices
    """
    # 1. Load CSV
    disease_df = pd.read_csv(csv_path)

    # 2. Parse textual columns
    text_columns = ["Risk Factors", "Symptoms", "Signs"]
    for col in text_columns:
        disease_df[col] = disease_df[col].apply(parse_string_list)
        disease_df[col] = disease_df[col].apply(lambda x: " ".join(x))

    # 3. Create TF-IDF feature matrix (one vectorizer per column)
    vectorizers = {}
    tfidf_parts = []
    for col in text_columns:
        vec = TfidfVectorizer()
        col_vec = vec.fit_transform(disease_df[col])
        vectorizers[col] = vec
        tfidf_parts.append(col_vec)

    # Combine horizontally
    tfidf_matrix = hstack(tfidf_parts)

    # 4. Encode disease labels
    if "Disease" in disease_df.columns:
        disease_labels = disease_df["Disease"].values
    else:
        # fallback if "Disease" not present
        disease_labels = disease_df["Subtype"].values

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(disease_labels)

    # 5. Train a default KNN model (with k=3 & metric='euclidean') 
    #    so we have a baseline model. We'll later re-train with user-specified k/metric.
    knn_model = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    knn_model.fit(tfidf_matrix, encoded_labels)

    return disease_df, tfidf_matrix, label_encoder, vectorizers, knn_model

# ------------------------------
# 3) App Layout & Logic
# ------------------------------
def main():
    st.title("Disease Classification Using KNN (TF-IDF)")

    # 3.1 Load data & pre-trained model
    disease_df, tfidf_matrix, label_encoder, vectorizers, default_knn = load_data()
    # Convert label array for easy usage
    if "Disease" in disease_df.columns:
        label_column = "Disease"
    else:
        label_column = "Subtype"
    disease_labels = label_encoder.transform(disease_df[label_column])

    # 3.2 Let user pick K and distance metric
    st.sidebar.title("KNN Configuration")
    k = st.sidebar.slider("Select K (number of neighbors)", min_value=1, max_value=10, value=3)
    metric = st.sidebar.selectbox("Select Distance Metric", ["euclidean", "manhattan", "cosine"])

    # 3.3 Build & train new KNN with user-selected parameters
    knn_model = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn_model.fit(tfidf_matrix, disease_labels)

    st.write(f"**Model configured with K={k} and metric='{metric}'**")
    st.write("Below is a preview of the first 5 rows of your dataset:")
    st.dataframe(disease_df.head())

    # 3.4 Accept user input for new data
    st.subheader("Predict a Disease from Your Input")

    st.write("Enter some text for each field. We'll vectorize them with the **same** TF-IDF vectorizers used in training.")

    # e.g. "fever, high bp"
    user_risk = st.text_input("Risk Factors (comma/space-separated)", value="")
    user_syms = st.text_input("Symptoms (comma/space-separated)", value="")
    user_signs = st.text_input("Signs (comma/space-separated)", value="")

    if st.button("Predict Disease"):
        # 1) Convert user text to single strings (just remove commas, for example)
        # e.g. "fever, high bp" -> "fever high bp"
        user_risk_str = user_risk.replace(",", " ")
        user_syms_str = user_syms.replace(",", " ")
        user_signs_str = user_signs.replace(",", " ")

        # 2) Transform each input using the stored vectorizers
        risk_vec  = vectorizers["Risk Factors"].transform([user_risk_str])
        syms_vec  = vectorizers["Symptoms"].transform([user_syms_str])
        signs_vec = vectorizers["Signs"].transform([user_signs_str])

        # 3) Combine horizontally
        user_features = hstack([risk_vec, syms_vec, signs_vec])

        # 4) Predict
        pred_label_encoded = knn_model.predict(user_features)[0]
        pred_label = label_encoder.inverse_transform([pred_label_encoded])[0]

        st.success(f"**Predicted Disease / Subtype:** {pred_label}")

if __name__ == "__main__":
    main()
