import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

def prueba_independencia_chi2(df, columna_objetivo):
    chi2_results = []
    for column in df.drop(columna_objetivo, axis=1).columns:
        tablas_contingencia = pd.crosstab(df[column], df[columna_objetivo])
        chi2, p, _, _ = chi2_contingency(tablas_contingencia)
        chi2_results.append((column, chi2, p))
    return pd.DataFrame(chi2_results, columns=['Caracteristica', 'Valor de Chi-cuadrado', 'p-value'])

def tablas_contingencia(df, columna_objetivo):
    contingency_tables = {}
    # Obtener las tablas de contingencia para cada columna categórica con respecto a la variable objetivo
    for column in df.drop(columna_objetivo, axis=1).columns:
        # Tabla de contingencia normalizada
        table = pd.crosstab(df[column], df[columna_objetivo], normalize='index')
        contingency_tables[column] = table
        plt.figure(figsize=(10, 6))
        sns.heatmap(table, annot=True, fmt='.2', cmap='coolwarm', cbar=True, linewidths=0.5, linecolor='black') 
        plt.title(f'Tabla de Contingencia: {column} vs {columna_objetivo}')
        plt.xlabel(columna_objetivo)
        plt.ylabel(column)
        plt.savefig(f'tabla_contingencia_proporciones_{column}.png', dpi=300, bbox_inches='tight')
        plt.close()

    return contingency_tables

def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    return np.sqrt(phi2corr / denom) if denom > 0 else np.nan

def cramers_v_variableObjetivo(df, columna_objetivo):
    results = []
    for col in df.drop(columna_objetivo, axis=1).columns:
        v = cramers_v(df[col], df[columna_objetivo])
        results.append((col, v))
    result_df = pd.DataFrame(results, columns=["Caracteristica", "Resultado Cramer V"]).sort_values(by="Resultado Cramer V", ascending=False)
    return result_df

def cramers_v_matrix(df, drop_columns=None):
    if drop_columns is not None:
        df = df.drop(columns=drop_columns)

    columnas_categoricas = df.select_dtypes(include='object').columns
    n = len(columnas_categoricas)
    result = pd.DataFrame(np.zeros((n, n)), index=columnas_categoricas, columns=columnas_categoricas)

    for col1 in columnas_categoricas:
        for col2 in columnas_categoricas:
            if col1 != col2:
                result.loc[col1, col2] = cramers_v(df[col1], df[col2])
            else:
                result.loc[col1, col2] = 1.0

    plt.figure(figsize=(12, 10))
    sns.heatmap(result, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de Cramér's V entre variables categóricas")
    plt.tight_layout()
    plt.savefig("cramers_v_matrix_completa.png")
    plt.show()

    return result

def carga_procesar_datos(filepath):
    # Codificar variable objetivo ('class') para que sea binaria, 
    # donde 'e' de edible se convierte en 0 y 'poisonous' de venenoso se convierte en 1
    df = pd.read_csv(filepath)
    
    df = df.drop(columns=['veil-type'])
    df['class'] = df['class'].map({'e': 0, 'p': 1})

    # Codificar el resto de variables categóricas con one-hot encoding
    X_dummies = pd.get_dummies(df.drop('class', axis=1))

    # Se define la variable objetivo 'class' y se normalizan las características
    y = df['class']
    X_normalizado = StandardScaler().fit_transform(X_dummies)

    # ------------------------- Seccion de análisis exploratorio de datos (EDA) -------------------------
    # Prueba de independencia de Chi-cuadrado para verificar la relación entre las características y la variable objetivo
    #chi2_results = prueba_independencia_chi2(df, 'class')
    #print("Resultados de la prueba Chi-cuadrado:")
    #print(chi2_results)

    # Correlacion de Cramér’s V para evaluar la relación entre las variables categóricas y la variable objetivo
    #cramers_df = cramers_v_variableObjetivo(df, 'class')
    #print("Matriz de Cramér’s V (relación con la variable objetivo):")
    #print(cramers_df)

    # Generar tablas de contingencia para cada característica con respecto a la variable objetivo
    #tablas_contingencia(df, 'class')

    #df = pd.read_csv("mushrooms.csv")
    #cramers_matrix = cramers_v_matrix(df, drop_columns=["veil-type"])
    #print(cramers_matrix)
    # ----------------------------------------------------------------------------------------------------

    # Dividir los datos en conjuntos de entrenamiento y prueba
    return train_test_split(X_normalizado, y, test_size=0.2, random_state=42)
