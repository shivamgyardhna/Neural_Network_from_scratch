import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, loss_activation, optimizer, epochs=1000):
        """
        model: list of layers and activations in forward order
        loss_activation: combined loss + activation (e.g. Softmax + CCE)
        optimizer: optimizer instance (Adam, SGD, etc.)
        epochs: number of training epochs
        """
        self.model = model
        self.loss_activation = loss_activation
        self.optimizer = optimizer
        self.epochs = epochs

        self.losses = []
        self.accuracies = []
        self.learning_rates = []

    def forward(self, x, y):
        output = x
        for layer in self.model:
            layer.forward(output)
            output = layer.output

        loss = self.loss_activation.forward(output, y)
        return loss, self.loss_activation.output

    def backward(self, y):
        self.loss_activation.backward(self.loss_activation.output, y)
        grad = self.loss_activation.grad_inputs

        for layer in reversed(self.model):
            layer.backward(grad)
            grad = layer.grad_inputs

    def compute_accuracy(self, predictions, y):
        preds = np.argmax(predictions, axis=1)

        if len(y.shape) == 2:
            y_true = np.argmax(y, axis=1)
        else:
            y_true = y

        return np.mean(preds == y_true)

    def update_params(self):
        self.optimizer.pre_update_params()
        self.learning_rates.append(self.optimizer.current_learning_rate)

        for layer in self.model:
            if hasattr(layer, "weights"):
                self.optimizer.update_params(layer)

        self.optimizer.post_update_params()

    def train(self, x, y):
        pbar = trange(self.epochs, desc="Training", unit="epoch")

        for epoch in pbar:
            loss, predictions = self.forward(x, y)
            accuracy = self.compute_accuracy(predictions, y)

            self.losses.append(loss)
            self.accuracies.append(accuracy)

            self.backward(y)
            self.update_params()

            pbar.set_postfix(
                loss=f"{loss:.4f}",
                acc=f"{accuracy:.4f}",
                lr=f"{self.optimizer.current_learning_rate:.4f}"
            )
            
    def plot_loss(self):
        plt.figure()
        plt.plot(self.losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()

    def plot_accuracy(self):
        plt.figure()
        plt.plot(self.accuracies)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy")
        plt.show()

    def plot_lr(self):
        plt.figure()
        plt.plot(self.learning_rates)
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.title("Training-Learning-Rate")
        plt.show()
    
    def plot_lossvsacc(self):
        epochs = range(1, len(self.losses) + 1)

        plt.figure()
        plt.plot(epochs, self.losses, label="Loss")
        plt.plot(epochs, self.accuracies, label="Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training Metrics")
        plt.legend()
        plt.show()
        
    def inference(self, x):
        output = x
        for layer in self.model:
            layer.forward(output)
            output = layer.output
        return output