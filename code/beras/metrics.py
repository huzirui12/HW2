import numpy as np

from beras.core import Callable



class CategoricalAccuracy(Callable):
    """
    Categorical accuracy metric.
    """

    def call(self, probs: Union[np.ndarray, Tensor], labels: Union[np.ndarray, Tensor]) -> float:
        """
        Compute and return the categorical accuracy of your model given the output probabilities
        and true labels.

        :param probs: Predicted probabilities, shape (batch_size, num_classes).
        :param labels: True labels, shape (batch_size,).
        :return: Categorical accuracy as a float.
        """
        if isinstance(probs, Tensor):
            probs = probs.numpy()
        if isinstance(labels, Tensor):
            labels = labels.numpy()

        # Convert probabilities to class predictions
        predictions = np.argmax(probs, axis=-1)

        # Compute categorical accuracy
        accuracy = np.mean(predictions == labels)

        return accuracy