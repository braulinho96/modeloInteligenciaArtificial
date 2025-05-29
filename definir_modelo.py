from keras.models import Sequential
from keras.layers import Dense, Input
import tensorflow as tf
from keras.regularizers import Regularizer 

# Funciones de penalidad
# Definición de clases de regularización personalizada que heredan de Regularizer

# Smoothed Absolute Regularizer
# Es una versión suavizada del L1 (que usa |x|), para evitar problemas en el punto no diferenciable en cero.
# Añadir epsilon dentro de la raíz cuadrada garantiza la diferenciabilidad.
# Penaliza pesos grandes, pero con una transición más suave.
# Ventaja: Evita explosiones de gradiente cerca de 0 y es útil para mantener algunos pesos pequeños sin hacerlos exactamente cero.
class SmoothedAbs(Regularizer):
    def __init__(self, lmbd=0.01, epsilon=1e-4):
        self.lmbd = lmbd
        self.epsilon = epsilon

    def __call__(self, x):
        return self.lmbd * tf.reduce_sum(tf.sqrt(tf.square(x) + self.epsilon))

    def get_config(self):
        return {'lmbd': self.lmbd, 'epsilon': self.epsilon}

# Rounded Square Regularizer
# Similar al L2, pero impone un límite superior (clip) a la penalización.
# Esto impide que los pesos con valores grandes sean castigados excesivamente.
# Ventaja: Previene que el modelo evite completamente pesos grandes si son necesarios, protegiendo la capacidad de aprendizaje en algunas neuronas.
class RoundedSquare(Regularizer):
    def __init__(self, lmbd=0.01, max_val=5.0):
        self.lmbd = lmbd
        self.max_val = max_val

    def __call__(self, x):
        squared = tf.square(x)
        clipped = tf.minimum(squared, self.max_val)
        return self.lmbd * tf.reduce_sum(clipped)

    def get_config(self):
        return {'lmbd': self.lmbd, 'max_val': self.max_val}
    
# Contrast Regularizer
# Penaliza valores intermedios de los pesos (cerca de 0.5 en módulo).
# El producto |x|(1 - |x|) tiene su pico en |x| = 0.5 y cae a 0 cuando x se acerca a 0 o 1.
# Fomenta pesos más binarios o extremos.
# Ventaja: Muy útil en modelos con muchas variables dummy (como los que salen de one-hot), ya que empuja al modelo a tomar decisiones firmes (activa o desactiva neuronas).
class ContrastRegularizer(Regularizer):
    def __init__(self, lmbd=0.01):
        self.lmbd = lmbd

    def __call__(self, x):
        return self.lmbd * tf.reduce_sum(tf.abs(x) * (1 - tf.abs(x)))

    def get_config(self):
        return {'lmbd': self.lmbd}
    
# Entropy-like Regularizer
# Inspirada en la entropía binaria.
# Penaliza valores de peso cercanos a 0.5 (alta incertidumbre).
# Empuja a que los pesos estén más cerca de 0 o 1, reduciendo la ambigüedad.
# Ventaja: Ideal para mejorar la interpretabilidad y evitar pesos indecisos, promoviendo una estructura más clara.
class EntropyLikeRegularizer(Regularizer):
    def __init__(self, lmbd=0.01, epsilon=1e-5):
        self.lmbd = lmbd
        self.epsilon = epsilon

    def __call__(self, x):
        x = tf.clip_by_value((x + 1) / 2, self.epsilon, 1 - self.epsilon)  # escala [-1,1] a [0,1]
        return -self.lmbd * tf.reduce_sum(x * tf.math.log(x) + (1 - x) * tf.math.log(1 - x))

    def get_config(self):
        return {'lmbd': self.lmbd, 'epsilon': self.epsilon}

# Variance Suppression Regularizer
# Penaliza la varianza entre los pesos.
# Intenta hacer que los pesos de una capa estén más uniformes entre sí.
# No busca reducir su valor absoluto, sino armonizar su dispersión.
# Ventaja: Útil para evitar que ciertas conexiones dominen la red y para inducir una regularización colectiva más equilibrada.
class VarianceSuppression(Regularizer):
    def __init__(self, lmbd=0.01):
        self.lmbd = lmbd

    def __call__(self, x):
        mean = tf.reduce_mean(x)
        return self.lmbd * tf.reduce_mean(tf.square(x - mean))

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