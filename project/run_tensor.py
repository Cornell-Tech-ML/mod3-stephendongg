"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

from turtle import forward
import minitorch

# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)
class Network(minitorch.Module):
    """
    A simple feedforward neural network with two hidden layers and ReLU activations.

    This network consists of three linear layers. The first two layers are followed
    by ReLU activation functions, and the final layer is followed by a sigmoid
    activation for binary classification tasks  .

    Attributes:
        layer1 (Linear): The first linear layer.
        layer2 (Linear): The second linear layer.
        layer3 (Linear): The third linear layer that outputs the final prediction.
    """

    def __init__(self, hidden_size):
        """
        Initializes the network's layers.

        Args:
            hidden_size (int): The number of neurons in the hidden layers.
        """
        super().__init__()
        self.layer1 = Linear(2, hidden_size)
        self.layer2 = Linear(hidden_size, hidden_size)
        self.layer3 = Linear(hidden_size, 1)

    def forward(self, x):
        """
        Defines the forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, 1) after passing through the network.
        """
        # First hidden layer with ReLU activation
        middle1 = self.layer1(x).relu()
        middle2 = self.layer2(middle1).relu()
        return self.layer3(middle2).sigmoid()


class Linear(minitorch.Module):
    """
    A linear (fully connected) layer for neural networks.

    The layer performs a linear transformation: y = Wx + b, where W is the weight matrix,
    x is the input, and b is the bias.

    Attributes:
        weights (RParam): The weight matrix.
        bias (RParam): The bias vector.
        out_size (int): The size of the output.
    """

    def __init__(self, in_size, out_size):
        """
        Initializes the Linear layer's weights and biases.

        Args:
            in_size (int): The size of the input.
            out_size (int): The size of the output.
        """
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        """
        Performs the forward pass through the linear layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_size) after the linear transformation.
        """
        batch_size, in_size = x.shape
        weights_reshaped = self.weights.value.view(1, in_size, self.out_size)
        x_reshaped = x.view(batch_size, in_size, 1)

        # Perform the matrix multiplication and add the bias
        output = weights_reshaped * x_reshaped
        result = output.sum(1).view(batch_size, self.out_size)

        # Add the bias and return the final output
        return result + self.bias.value





def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
