import numpy as np
from My_NN.activation import Softmax
from My_NN.activation import ReLU


# Common loss class
class Loss:

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss


class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples), y_true
            ]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods


    def backward(self, dvalues, y_true):

        # Number of samples
        samp = len(y_true)

        labels = len(dvalues[0])

        if y_true.ndim == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = y_true / dvalues
        self.dinputs = self.dinputs / samp


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Softmax_Loss_CategoricalCrossentropy:

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Softmax()
        self.loss = Loss_CategoricalCrossentropy()


    # Forward pass
    def forward(self, inputs, y_true):

        # Output layer's activation function
        self.activation.forward(inputs)

        self.inputs = inputs.copy()

        # Set the output
        self.output = self.activation.output

        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)


    def backward(self, dvalues, y_true):

        # Number of samples
        samp = len(y_true)

        labels = len(dvalues[0])

        if y_true.ndim == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = dvalues.copy()
        self.dinputs = self.dinputs - y_true
        self.dinputs = self.dinputs / samp