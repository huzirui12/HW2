import numpy as np
import tensorflow as tf
from beras.core import Diffable, Tensor


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    """
    TODO:
        - call function
        - input_gradients
    Identical to HW1!
    """
    def call(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        TODO: Find the Mean Squared Error of y_pred and y_true

        y_pred: the predicted labels
        y_true: the true labels
        returns: the MeanSquaredError as a Tensor
        """
        self.y_pred = y_pred
        self.y_true = y_true
        mse = np.mean((y_pred - y_true) ** 2)
        return mse

    def get_input_gradients(self) -> list[Tensor]:
        """
        TODO: Return the gradients of the layer in the same order as the inputs of call
        i.e. return the gradient of the layer w.r.t y_pred, the gradient of the layer w.r.t. y_true

        returns: a list of input gradients in the same order as the input arguments of the call function.
        HINT: What would the gradients be with respect to a scalar?
        """
        # Compute the gradient of the loss with respect to y_pred
        grad_y_pred = (2 / len(self.y_pred)) * (self.y_pred - self.y_true)

        # Compute the gradient of the loss with respect to y_true
        grad_y_true = -(2 / len(self.y_true)) * (self.y_pred - self.y_true)

        return [grad_y_pred, grad_y_true]


def clip_0_1(x, eps=1e-8):
    return np.clip(x, eps, 1 - eps)



class CategoricalCrossentropy(Loss):
    """
    Categorical Crossentropy loss function.
    """

    def call(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)  # Clip values to stabilize calculations
        return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        raise NotImplementedError("Input gradients for CategoricalCrossentropy loss are not supported.")