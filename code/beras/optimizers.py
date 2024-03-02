from collections import defaultdict
import numpy as np

"""
TODO: Implement all the apply_gradients for the 3 optimizers:
    - BasicOptimizer
    - RMSProp
    - Adam
"""

class BasicOptimizer:
    """
    This class represents a basic optimizer which simply applies the scaled gradients to the weights.

    TODO: Roadmap 5.
        - apply_gradients 
    """
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def apply_gradients(self, weights, grads):
        for weight, grad in zip(weights, grads):
            if grad is None:
                print(f"Gradient for weight {weight} is None")  # Debugging print
                continue  # Skip applying gradient if it's None
            weight -= self.learning_rate * grad



class RMSProp:
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = defaultdict(lambda: 0)

    def apply_gradients(self, weights, grads):
        """
        Apply RMSProp optimization to update weights.

        :param weights: Dictionary containing weights.
        :param grads: Dictionary containing gradients.
        """
        for weight_key, grad_value in grads.items():
            self.v[weight_key] = self.beta * self.v[weight_key] + (1 - self.beta) * grad_value ** 2
            weights[weight_key] -= (self.learning_rate / (self.v[weight_key] ** 0.5 + self.epsilon)) * grad_value

class Adam:
    def __init__(
        self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    ):
        self.amsgrad = amsgrad

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = defaultdict(lambda: 0)         # First moment zero vector
        self.v = defaultdict(lambda: 0)         # Second moment zero vector.
        self.m_hat = defaultdict(lambda: 0)     # Expected value of first moment vector
        self.v_hat = defaultdict(lambda: 0)     # Expected value of second moment vector
        self.t = 0                              # Time counter

    def apply_gradients(self, weights, grads):
        """
        Apply Adam optimization to update weights.

        :param weights: Dictionary containing weights.
        :param grads: Dictionary containing gradients.
        """
        self.t += 1

        for weight_key, grad_value in grads.items():
            # Update biased first moment estimate
            self.m[weight_key] = self.beta_1 * self.m[weight_key] + (1 - self.beta_1) * grad_value
            # Update biased second moment estimate
            self.v[weight_key] = self.beta_2 * self.v[weight_key] + (1 - self.beta_2) * grad_value ** 2

            # Compute bias-corrected first moment estimate
            self.m_hat[weight_key] = self.m[weight_key] / (1 - self.beta_1 ** self.t)
            # Compute bias-corrected second moment estimate
            self.v_hat[weight_key] = self.v[weight_key] / (1 - self.beta_2 ** self.t)

            # Compute the step size
            step = self.learning_rate / (self.v_hat[weight_key] ** 0.5 + self.epsilon)

            # Update weights
            weights[weight_key] -= step * self.m_hat[weight_key]