import PySimpleGUI as sg
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

def predict_heart_disease(model, patient_data):
    patient_df = pd.DataFrame([patient_data])
    probability = model.predict_proba(patient_df)[0][1]
    prediction = model.predict(patient_df)[0]
    return probability, prediction

def format_prediction(probability, prediction):
    risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"
    result = {
        "Prediction": "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected",
        "Probability": f"{probability:.1%}",
        "Risk Level": risk_level
    }
    return result


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

