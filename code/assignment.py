from types import SimpleNamespace
 
import beras
import numpy as np

"""
TODO:
    - Implement a sequential model
    - Finish get_simple_model_components
    - Finish get_advanced_model_components
    - Run the file to train your models!
"""
class SequentialModel(beras.Model):
    """
    A sequential model class inherited from beras.Model.
    """

def call(self, inputs: beras.Tensor) -> beras.Tensor:
        """
        Forward pass in sequential model.

        :param inputs: Input tensor.
        :return: Output tensor.
        """
        output = inputs
        for layer in self.layers:
            output = layer(output)
        return output

def batch_step(self, x, y, training=True) -> dict[str, float]:
        """
        Compute loss and accuracy for a batch.

        :param x: Input data.
        :param y: Target labels.
        :param training: Whether the model is being trained.
        :return: Dictionary containing loss and accuracy.
        """
        with beras.GradientTape() as tape:
            predictions = self.call(x)
            loss = self.compiled_loss(y, predictions)

        if training:
            gradients = tape.gradient(loss, self.weights)
            self.optimizer.apply_gradients(zip(gradients, self.weights))

        accuracy = self.compiled_acc(y, predictions)

        return {"loss": loss, "accuracy": accuracy}

def get_simplest_model_components() -> SimpleNamespace:
    """
    Returns a simple single-layer model. You can try running this one
    as a first test if you'd like. This one will not be evaluated though.

    :return: model
    """
    from beras.layers import Dense
    from beras.losses import MeanSquaredError
    from beras.metrics import CategoricalAccuracy
    from beras.optimizers import BasicOptimizer


    model = SequentialModel([
        Dense(784, 10, initializer="normal"),
    ])
    model.compile(
        optimizer=BasicOptimizer(0.1),
        loss_fn=MeanSquaredError(),
        acc_fn=CategoricalAccuracy(),
    )
    return SimpleNamespace(model=model, epochs=10, batch_size=256)


def get_simple_model_components() -> SimpleNamespace:
    """
    Returns components for a simple single-layer model.

    :return: SimpleNamespace containing model, epochs, and batch_size.
    """
    # Define the model
    model = SequentialModel([
        Dense(units=64, activation=LeakyReLU(alpha=0.3)),
        Dense(units=10, activation=Sigmoid())
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss_fn=MeanSquaredError(),
        acc_fn=CategoricalAccuracy()
    )

    # Return the model components
    return SimpleNamespace(model=model, epochs=10, batch_size=256)

def get_advanced_model_components() -> SimpleNamespace:
    """
    Returns a multi-layered model with more involved components.
    """

    # TODO: Implement a similar model, but make sure to use Softmax and CategoricalCrossentropy
    # model = ?

    from beras.activations import ReLU, LeakyReLU, Softmax
    from beras.layers import Dense
    from beras.losses import CategoricalCrossentropy, MeanSquaredError
    from beras.metrics import CategoricalAccuracy
    from beras.optimizers import Adam

    # Define the model
    model = SequentialModel([
        Dense(units=128, activation=ReLU()),
        Dense(units=64, activation=ReLU()),
        Dense(units=10, activation=Softmax())
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss_fn=CategoricalCrossentropy(),
        acc_fn=CategoricalAccuracy()
    )

    # Return the model components
    return SimpleNamespace(model=model, epochs=10, batch_size=256)


if __name__ == "__main__":
    """
    Read in MNIST data, initialize your model, and train and test your model.
    """
    from beras.onehot import OneHotEncoder
    from preprocess import load_and_preprocess_data

    ## Read in MNIST data,
    train_inputs, train_labels, test_inputs, test_labels = load_and_preprocess_data()

    ## Read in MNIST data, use the OneHotEncoder class to one hot encode the labels,
    ## instantiate and compile your model, and train your model
    ohe = OneHotEncoder()
    concat_labels = np.concatenate([train_labels, test_labels], axis=-1)
    ohe.fit(concat_labels)

    ## Threshold of accuracy: 
    ##  >95% on testing accuracy from get_simple_model_components
    ##  >95% on testing accuracy from get_advanced_model_components
    arg_comps = [
        get_simplest_model_components(),  ## Simple starter option, similar to HW1. Not graded
        # get_simple_model_components(),    ## Simple model using sigmoid and MSE; >95% accuracy
        # get_advanced_model_components()   ## Advanced model using softmax and CCE; >95% accuracy
    ]
    for args in arg_comps:

        train_agg_metrics = args.model.fit(
            train_inputs,
            ohe(train_labels),
            epochs=args.epochs,
            batch_size=args.batch_size
        )

        ## Feel free to use the visualize_metrics function to view your accuracy and loss.
        ## The final accuracy returned during evaluation must be > 80%.

        # from visualize import visualize_images, visualize_metrics
        # visualize_metrics(train_agg_metrics["loss"], train_agg_metrics["acc"])
        # visualize_images(model, train_inputs, ohe(train_labels))

        test_agg_metrics = args.model.evaluate(test_inputs, ohe(test_labels), batch_size=100)
        print("Testing Performance:", test_agg_metrics)
