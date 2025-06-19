import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

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

    # Variables categóricas a analizar
    variables = [
        'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
        'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
        'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
    ]

    # Generar tabla de contingencia para cada variable
    for var in variables:
        # Tabla de frecuencias (no normalizada)
        contingency = pd.crosstab(df[var], df['class'])

        # Prueba de chi-cuadrado
        chi2, p, dof, expected = chi2_contingency(contingency)

        print(f"\nVariable: {var}")
        print(f"Chi-cuadrado = {chi2:.4f}, p-valor = {p:.7f}, grados de libertad = {dof}")

        if p < 0.05:
            print("↳ Existe una asociación significativa con la clase.")
        else:
            print("↳ No hay asociación significativa con la clase.")
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    return train_test_split(X, y, test_size=0.2, random_state=42)
