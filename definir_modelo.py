from keras.models import Sequential
from keras.layers import Dense, Input
import tensorflow as tf
from keras.regularizers import Regularizer 
  
# Variance Suppression Regularizer
# Penaliza la varianza entre los pesos.
# Intenta hacer que los pesos de una capa estén más uniformes entre sí.
# No busca reducir su valor absoluto, sino armonizar su dispersión.
# Ventaja: Útil para evitar que ciertas conexiones dominen la red y para inducir una regularización colectiva más equilibrada.
# Desventaja: Puede ser menos efectivo si los pesos necesitan variar ampliamente para capturar la complejidad de los datos.

class VarianceSuppression(Regularizer):
    def __init__(self, lmbd=0.001):
        self.lmbd = lmbd

    def __call__(self, x):
        mean = tf.reduce_mean(x)
        return self.lmbd * tf.reduce_mean(tf.square(x - mean))

    def get_config(self):
        return {'lmbd': self.lmbd}

# Max Penalty Regularizer
# Penaliza el valor máximo absoluto de los pesos.
# Esto puede ayudar a evitar que algunos pesos se vuelvan excesivamente grandes, lo que podría causar problemas de estabilidad en el entrenamiento.
# # Ventaja: Ayuda a controlar los pesos más grandes, lo que puede ser útil para evitar problemas de sobreajuste o inestabilidad en el entrenamiento.
# # Desventaja: Puede ser menos efectivo si los pesos necesitan variar ampliamente para capturar la complejidad de los datos.
class MaxPenaltyRegularizer(Regularizer):
    def __init__(self, lmbd=0.001):
        self.lmbd = lmbd

    def __call__(self, x):
        return self.lmbd * tf.reduce_max(tf.abs(x))

    def get_config(self):
        return {'lmbd': self.lmbd}

# Definición del modelo de Keras con regularización personalizada
# Se define una función que construye el modelo con las capas y se le aplican los regularizadores personalizados
#   Se utiliza Sequential() para construir el modelo de forma secuencial, añadiendo capas densas con activación ReLU y una capa de salida con activación sigmoide para la clasificación binaria.
#   Relu: es una función de activación que introduce no linealidades en el modelo, permitiendo que aprenda patrones complejos.
#   Sigmoide: es una función de activación que se utiliza en la capa de salida para la clasificación binaria, ya que produce una probabilidad entre 0 y 1.

def construir_modelo(input_dim, regularizer_instance):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizer_instance))
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizer_instance))
    model.add(Dense(1, activation='sigmoid')) 
    return model