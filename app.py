import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef
)
import seaborn as sns
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="Obesity Classification App")

st.title("Obesity Level Classification")
st.write("Upload test.csv file to predict obesity levels.")

st.subheader("Download Sample Test File")

with open("test.csv", "rb") as file:
    st.download_button(
        label="Download test.csv",
        data=file,
        file_name="test.csv",
        mime="text/csv"
    )

# Model Selection
model_options = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "K-Nearest Neighbors": "model/k-nearest_neighbors.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

selected_model_name = st.selectbox("Select Model", list(model_options.keys()))
MODEL_PATH = model_options[selected_model_name]

# Load Saved Objects
SCALER_PATH = "model/scaler.pkl"
FEATURE_ENCODER_PATH = "model/feature_label_encoders.pkl"
TARGET_ENCODER_PATH = "model/target_label_encoder.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(FEATURE_ENCODER_PATH, "rb") as f:
    feature_encoders = pickle.load(f)

with open(TARGET_ENCODER_PATH, "rb") as f:
    target_encoder = pickle.load(f)

# File Upload
uploaded_file = st.file_uploader("Upload test.csv file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # Handle Target Column
    if "NObeyesdad" in df.columns:
        y_true = df["NObeyesdad"]
        y_true_encoded = target_encoder.transform(y_true)
        df = df.drop(columns=["NObeyesdad"])
    else:
        y_true_encoded = None

    # Keeping copy for results display
    result_df = df.copy()

    # Preprocessing
    categorical_cols = list(feature_encoders.keys())
    numerical_cols = [col for col in df.columns if col not in categorical_cols]

    # Encode categorical features
    for col in categorical_cols:
        df[col] = feature_encoders[col].transform(df[col])

    # Scale numerical features
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    df = df[model.feature_names_in_]

    # Prediction
    predictions = model.predict(df)
    decoded_predictions = target_encoder.inverse_transform(predictions)

    result_df["Predicted_Obesity_Level"] = decoded_predictions

    st.subheader("Prediction Results")
    st.dataframe(result_df.head())

    # Evaluation Section
    if y_true_encoded is not None:

        st.subheader("Evaluation Metrics")

        accuracy = accuracy_score(y_true_encoded, predictions)
        precision = precision_score(y_true_encoded, predictions, average="macro")
        recall = recall_score(y_true_encoded, predictions, average="macro")
        f1 = f1_score(y_true_encoded, predictions, average="macro")
        mcc = matthews_corrcoef(y_true_encoded, predictions)

        # AUC (only if model supports probabilities)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(df)
            auc = roc_auc_score(
                y_true_encoded,
                y_prob,
                multi_class="ovr",
                average="macro"
            )
        else:
            auc = None

        st.write(f"**Accuracy:** {accuracy:.4f}")

        if auc is not None:
            st.write(f"**AUC (Macro OVR):** {auc:.4f}")

        st.write(f"**Precision (Macro):** {precision:.4f}")
        st.write(f"**Recall (Macro):** {recall:.4f}")
        st.write(f"**F1 Score (Macro):** {f1:.4f}")
        st.write(f"**MCC:** {mcc:.4f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true_encoded, predictions)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_,
            ax=ax
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)