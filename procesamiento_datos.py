import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    # Cargar datos
    df = pd.read_csv(filepath)

    # Eliminar columnas innecesarias
    df.drop(['customerID'], axis=1, inplace=True)

    # Convertir 'TotalCharges' a numérico
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # Codificar variables categóricas
    for column in df.select_dtypes(include=['object']).columns:
        if column != 'Churn':
            df[column] = LabelEncoder().fit_transform(df[column])

    # Codificar variable objetivo
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    # Dividir en características y objetivo
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Escalar características
    X = StandardScaler().fit_transform(X)

    # Dividir en conjuntos de entrenamiento y prueba
    return train_test_split(X, y, test_size=0.2, random_state=42)
