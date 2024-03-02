import numpy as np

from typing import Literal
from beras.core import Diffable, Variable, Tensor

DENSE_INITIALIZERS = Literal["zero", "normal", "xavier", "kaiming", "xavier uniform", "kaiming uniform"]

class Dense(Diffable):
    """
    This class represents a Dense (or Fully Connected) layer.

    TODO: Roadmap 2.
        - weights
        - call
        - get_input_gradients
        - get_weight_gradients
        - _initialize_weights
    """

    def __init__(self, input_size, output_size, initializer: DENSE_INITIALIZERS = "normal"):
        self.input = None
        self.w, self.b = self._initialize_weight(initializer, input_size, output_size)

    @property
    def weights(self) -> list[Tensor]:
        """
        TODO: return the weights (and biases) of this dense layer
        Hint: check out the Dense layer's instance variables as defined in the constructor __init__

        returns: the weights (and biases) of this Dense Layer
        """
        return [self.w, self.b]

    def call(self, x: Tensor) -> Tensor:
        """
        TODO: Forward pass for a dense layer! Refer to lecture slides for how this is computed.

        x: input data of shape [num_samples, input_size]
        returns: the forward pass of the dense layer performed on x
        """
        self.input = x
        output = np.dot(x, self.w) + self.b
        return output

    def get_input_gradients(self) -> list[Tensor]:
        """
        TODO: Return the gradient of this layer with respect to its input, as a list
        You should have as many gradients as inputs (just one)

        returns: a list of gradients in the same order as its inputs
        """
        input_gradient = self.w.T
        return [input_gradient]

    def get_weight_gradients(self, output_gradient: Tensor) -> list[Tensor]:
        if self.input is None:
            raise ValueError("Input must not be None. Ensure 'call' method was called before this function.")
        grad_w = np.dot(self.input.T, output_gradient)  # Gradient w.r.t weights
        grad_b = np.sum(output_gradient, axis=0, keepdims=True)  # Gradient w.r.t biases
        return [grad_w, grad_b]

    @staticmethod
    def _initialize_weight(initializer, input_size, output_size) -> tuple[Variable, Variable]:
        """
        TODO: return the initialized weight, bias Variables as according to the initializer.

        initializer: string representing which initializer to use. see below for details
        input_size: size of latent dimension of input
        output_size: size of latent dimension of output
        returns: weight, bias as **Variable**s.

        Initializes the values of the weights and biases. The bias weights should always start at zero.
        However, the weights should follow the given distribution defined by the initializer parameter
        (zero, normal, xavier, or kaiming). You can do this with an if statement
        cycling through each option!

        Details on each weight initialization option:
            - Zero: Weights and biases contain only 0's. Generally a bad idea since the gradient update
            will be the same for each weight so all weights will have the same values.
            - Normal: Weights are initialized according to a normal distribution.
            - Xavier: Goal is to initialize the weights so that the variance of the activations are the
            same across every layer. This helps to prevent exploding or vanishing gradients. Typically
            works better for layers with tanh or sigmoid activation.
            - Kaiming: Similar purpose as Xavier initialization. Typically works better for layers
            with ReLU activation.
        """

        initializer = initializer.lower()
        assert initializer in (
            "zero",
            "normal",
            "xavier",
            "kaiming",
            "xavier uniform",
            "kaiming uniform",
        ), f"Unknown dense weight initialization strategy '{initializer}' requested"

        #io_size = (input_size, output_size)
        # HINT: self.w and self.b are both 2-dimensional tensors
        
        #pass ### Replace with your code!!
        if initializer == "zero":
            w = np.zeros((input_size, output_size))
            b = np.zeros(output_size)
        elif initializer == "normal":
            w = np.random.randn(input_size, output_size) * 0.01  # Small random numbers from normal distribution
            b = np.zeros(output_size)
        elif initializer in ("xavier", "xavier uniform"):
            scale = np.sqrt(6 / (input_size + output_size)) if initializer == "xavier uniform" else np.sqrt(2 / (input_size + output_size))
            w = np.random.uniform(-scale, scale, (input_size, output_size)) if initializer == "xavier uniform" else np.random.randn(input_size, output_size) * scale
            b = np.zeros(output_size)
        elif initializer in ("kaiming", "kaiming uniform"):
            scale = np.sqrt(6 / input_size) if initializer == "kaiming uniform" else np.sqrt(2 / input_size)
            w = np.random.uniform(-scale, scale, (input_size, output_size)) if initializer == "kaiming uniform" else np.random.randn(input_size, output_size) * scale
            b = np.zeros(output_size)

    # Wrap the numpy arrays w and b in Variable instances
        w_var = Variable(w)
        b_var = Variable(b)

        return w_var, b_var   