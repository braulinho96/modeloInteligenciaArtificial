from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
from tensorflow.keras.regularizers import Regularizer 

# Funciones de penalidad
def l1_penalty_func(W, lmbd):
    return lmbd * tf.reduce_sum(tf.abs(W))

def l2_penalty_func(W, lmbd): 
    return lmbd * tf.reduce_sum(tf.square(W))

# Se crean clases de regularizadores personalizados que heredan de Regularizer
class RegularizadorL1(Regularizer):
    def __init__(self, lmbd):
        self.lmbd = lmbd

    def __call__(self, weight_matrix):
        return l1_penalty_func(weight_matrix, self.lmbd)

    def get_config(self):
        return {'lmbd': self.lmbd}

# Clase de regularizador L2 personalizada
class RegularizadorL2(Regularizer):
    def __init__(self, lmbd):
        self.lmbd = lmbd

    def __call__(self, weight_matrix):
        return l2_penalty_func(weight_matrix, self.lmbd)

    def get_config(self):
        return {'lmbd': self.lmbd}
    


# Definición del modelo de Keras con regularización personalizada
# Se define una función que construye el modelo con las capas y se le aplican los regularizadores personalizados
# Se utiliza Sequential para construir el modelo de forma secuencial, añadiendo capas densas con activación ReLU y una capa de salida con activación sigmoide para la clasificación binaria.

# Relu: es una función de activación que introduce no linealidades en el modelo, permitiendo que aprenda patrones complejos.
# Sigmoide: es una función de activación que se utiliza en la capa de salida para la clasificación binaria, ya que produce una probabilidad entre 0 y 1.

def construir_modelo(input_dim, regularizer_instance):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizer_instance))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizer_instance))
    model.add(Dense(1, activation='sigmoid')) # Sigmoide para la clasificacion binaria
    return model