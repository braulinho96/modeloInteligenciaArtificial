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
    def __init__(self, lmbd=0.001, epsilon=1e-4):
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
    def __init__(self, lmbd=0.001, max_val=5.0):
        self.lmbd = lmbd
        self.max_val = max_val

    def __call__(self, x):
        squared = tf.square(x)
        clipped = tf.minimum(squared, self.max_val)
        return self.lmbd * tf.reduce_sum(clipped)

    def get_config(self):
        return {'lmbd': self.lmbd, 'max_val': self.max_val}
    
# Variance Suppression Regularizer
# Penaliza la varianza entre los pesos.
# Intenta hacer que los pesos de una capa estén más uniformes entre sí.
# No busca reducir su valor absoluto, sino armonizar su dispersión.
# Ventaja: Útil para evitar que ciertas conexiones dominen la red y para inducir una regularización colectiva más equilibrada.
class VarianceSuppression(Regularizer):
    def __init__(self, lmbd=0.001):
        self.lmbd = lmbd

    def __call__(self, x):
        mean = tf.reduce_mean(x)
        return self.lmbd * tf.reduce_mean(tf.square(x - mean))

    def get_config(self):
        return {'lmbd': self.lmbd}
    
class CosineRegularizer(Regularizer):
    def __init__(self, lmbd=0.001):
        self.lmbd = lmbd

    def __call__(self, x):
        return self.lmbd * tf.reduce_sum(1.0 - tf.cos(x))

    def get_config(self):
        return {'lmbd': self.lmbd}

class MaxPenaltyRegularizer(Regularizer):
    def __init__(self, lmbd=0.001):
        self.lmbd = lmbd

    def __call__(self, x):
        return self.lmbd * tf.reduce_max(tf.abs(x))

    def get_config(self):
        return {'lmbd': self.lmbd}

class SmoothStepRegularizer(Regularizer):
    def __init__(self, lmbd=0.001):
        self.lmbd = lmbd

    def __call__(self, x):
        smooth_step = tf.pow(x, 2) * (3 - 2 * tf.abs(x))  # Suave entre -1 y 1
        return self.lmbd * tf.reduce_sum(smooth_step)

    def get_config(self):
        return {'lmbd': self.lmbd}

# ----------------------------------------------------
class WeightOscillationDampener(Regularizer):
    def __init__(self, lmbd=0.001):
        self.lmbd = lmbd

    def __call__(self, x):
        return self.lmbd * tf.reduce_sum(tf.square(tf.sin(x)))

    def get_config(self):
        return {'lmbd': self.lmbd}

class MinimalEnergyRegularizer(Regularizer):
    def __init__(self, lmbd=0.001, target_energy=1.0):
        self.lmbd = lmbd
        self.target_energy = target_energy

    def __call__(self, x):
        energy = tf.reduce_sum(tf.square(x))
        return self.lmbd * tf.square(energy - self.target_energy)

    def get_config(self):
        return {'lmbd': self.lmbd, 'target_energy': self.target_energy}

class CenteredWeightRegularizer(Regularizer):
    def __init__(self, lmbd=0.001, center=0.1):
        self.lmbd = lmbd
        self.center = center

    def __call__(self, x):
        return self.lmbd * tf.reduce_sum(tf.square(x - self.center))

    def get_config(self):
        return {'lmbd': self.lmbd, 'center': self.center}

class EntropyLikeWeightRegularizer(Regularizer):
    def __init__(self, lmbd=0.001, epsilon=1e-7):
        self.lmbd = lmbd
        self.epsilon = epsilon

    def __call__(self, x):
        abs_x = tf.abs(x)
        probs = abs_x / (tf.reduce_sum(abs_x) + self.epsilon)
        entropy = -tf.reduce_sum(probs * tf.math.log(probs + self.epsilon))
        return self.lmbd * entropy

    def get_config(self):
        return {'lmbd': self.lmbd, 'epsilon': self.epsilon}

class AntiSaturationRegularizer(Regularizer):
    def __init__(self, lmbd=0.001):
        self.lmbd = lmbd

    def __call__(self, x):
        return self.lmbd * tf.reduce_sum(1.0 / (1.0 + tf.square(x)))

    def get_config(self):
        return {'lmbd': self.lmbd}

class SparseGroupRegularizer(Regularizer):
    def __init__(self, lmbd=0.001):
        self.lmbd = lmbd

    def __call__(self, x):
        group_norm = tf.sqrt(tf.reduce_sum(tf.square(x)))
        return self.lmbd * group_norm

    def get_config(self):
        return {'lmbd': self.lmbd}

class LayerSmoothnessRegularizer(Regularizer):
    def __init__(self, lmbd=0.001):
        self.lmbd = lmbd

    def __call__(self, x):
        diff = x[1:] - x[:-1]
        return self.lmbd * tf.reduce_sum(tf.square(diff))

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