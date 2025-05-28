import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    # Cargar datos
    df = pd.read_csv(filepath)

    # Eliminar columnas innecesarias
    #df.drop(['customerID'], axis=1, inplace=True)

    # Codificar variable objetivo ('class')
    df['class'] = df['class'].map({'e': 0, 'p': 1})

    # Codificar el resto de variables categóricas con one-hot encoding
    X = pd.get_dummies(df.drop('class', axis=1))

    y = df['class']

    # Escalar características
    X = StandardScaler().fit_transform(X)

    # Dividir en conjuntos de entrenamiento y prueba
    return train_test_split(X, y, test_size=0.2, random_state=42)
