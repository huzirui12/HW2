from collections import defaultdict

from beras.core import Diffable, Tensor

class GradientTape:

    def __init__(self):
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: defaultdict[int, Diffable | None] = defaultdict(lambda: None)

    def __enter__(self):
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        To find what methods are available on certain objects, reference the cheat sheet
        """
        queue = [target]                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}

        ### TODO: Populate the grads dictionary with {weight_id, weight_gradient} pairs.
        while queue:
            tensor = queue.pop(0)
            tensor_id = id(tensor)

            if tensor_id not in grads:
                # Compute gradient for the tensor
                if tensor_id in self.previous_layers:
                    prev_layer = self.previous_layers[tensor_id]
                    if prev_layer is not None:
                        # Compute the gradient using chain rule
                        prev_grads = prev_layer._backward(tensor, grad=grads[tensor_id])
                        # Store gradients for the previous layer's inputs
                        prev_inputs = prev_layer._get_inputs()
                        for prev_input, prev_grad in zip(prev_inputs, prev_grads):
                            prev_input_id = id(prev_input)
                            if grads[prev_input_id] is None:
                                grads[prev_input_id] = prev_grad
                            else:
                                grads[prev_input_id] += prev_grad

                        # Add previous layer's inputs to the queue for further propagation
                        queue.extend(prev_inputs)

            grads[tensor_id] = tensor.gradient if tensor_id == id(target) else grads[tensor_id]

        ## Retrieve the sources and make sure that all of the sources have been reached
        out_grads = [grads[id(source)][0] for source in sources]
        disconnected = [f"var{i}" for i, grad in enumerate(out_grads) if grad is None]

        if disconnected:
            print(f"Warning: The following tensors are disconnected from the target graph: {disconnected}")

        return out_grads