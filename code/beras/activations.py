import numpy as np

from .core import Diffable,Tensor

class Activation(Diffable):
    @property
    def weights(self): return []

    def get_weight_gradients(self): return []


################################################################################
## Intermediate Activations To Put Between Layers

class LeakyReLU(Activation):
    """
    Leaky ReLU activation function.
    """

    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha

    def call(self, x) -> Tensor:
        """Leaky ReLu forward propagation! """
        return tf.where(x >= 0, x, self.alpha * x)

    def get_input_gradients(self) -> list[Tensor]:
        """
        Leaky ReLu backpropagation!
        """
        return [tf.where(x >= 0, 1.0, self.alpha) for x in self._input]

    def compose_input_gradients(self, J):
        return [grad * J for grad in self.get_input_gradients()]


class ReLU(LeakyReLU):
    ## GIVEN: Just shows that relu is a degenerate case of the LeakyReLU
    def __init__(self):
        super().__init__(alpha=0)


################################################################################
## Output Activations For Probability-Space Outputs

class Sigmoid(Activation):
    """
    Sigmoid activation function.
    """

    def call(self, x) -> Tensor:
        """Sigmoid forward propagation!"""
        return tf.math.sigmoid(x)

    def get_input_gradients(self) -> list[Tensor]:
        """
        Sigmoid input gradients!
        """
        sigmoid = self.call(self._input[0])
        return [sigmoid * (1 - sigmoid)]

    def compose_input_gradients(self, J):
        return [grad * J for grad in self.get_input_gradients()]


class Softmax(Activation):
    """
    Softmax activation function.
    """

    def call(self, x):
        """Softmax forward propagation!"""
        return tf.nn.softmax(x)

    def get_input_gradients(self):
        """Softmax input gradients!"""
        softmax = self.call(self._input[0])
        n = tf.shape(softmax)[-1]
        # Reshape softmax to (..., 1, n)
        softmax = tf.expand_dims(softmax, axis=-2)
        # Compute Jacobian matrix
        J = -tf.matmul(softmax, tf.transpose(softmax, perm=[0, 2, 1]))
        # Fill diagonal elements
        J += tf.linalg.diagflat(softmax[..., 0])
        return [J]

    def compose_input_gradients(self, J):
        return [tf.matmul(J, grad) for grad in self.get_input_gradients()]
