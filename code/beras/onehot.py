import numpy as np

from beras.core import Callable

class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - Keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def __init__(self):
        self.label_dict = None

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        # Fetch all the unique labels and create a dictionary with
        # the unique labels as keys and their one hot encodings as values
        unique_labels = np.unique(data)
        self.label_dict = {label: np.eye(len(unique_labels))[i] for i, label in enumerate(unique_labels)}

    def call(self, data):
        """
        One-hot encodes the input data based on the fitted encoder.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        :return: 2D array containing one-hot encoded labels.
        """
        if self.label_dict is None:
            raise ValueError("Fit method must be called before calling the OneHotEncoder.")
        return np.array([self.label_dict[label] for label in data])

    def inverse(self, data):
        """
        Converts one-hot encoded labels back to their original representation.

        :param data: 2D array containing one-hot encoded labels.
        :return: 1D array containing original labels.
        """
        if self.label_dict is None:
            raise ValueError("Fit method must be called before calling the OneHotEncoder.")
        inv_label_dict = {tuple(encoded_label): label for label, encoded_label in self.label_dict.items()}
        return np.array([inv_label_dict[tuple(encoded_label)] for encoded_label in data])
        return np.array([inv_label_dict[tuple(encoded_label)] for encoded_label in data])