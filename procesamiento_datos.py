import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    # Cargar datos
    df = pd.read_csv(filepath)

    # Codificar variable objetivo ('class') para que sea binaria
    df['class'] = df['class'].map({'e': 0, 'p': 1})

    # Codificar el resto de variables categóricas con one-hot encoding
    # Crea columnas para cada categoría y elimina la columna original
    X = pd.get_dummies(df.drop('class', axis=1))

    # Se define la variable objetivo 'class'
    y = df['class']

    # Normalizar las características
    X = StandardScaler().fit_transform(X)
    
    # Dividir en conjuntos de entrenamiento y prueba
    return train_test_split(X, y, test_size=0.2, random_state=42)
