# Machine learning framework
import torch
# Interface for Matplotlib
import matplotlib.pyplot as plt

# Graph plotting library
import matplotlib
matplotlib.rcParams['figure.figsize'] = (13.0, 5.0)


# Function to be predicted by the neural network
def target_function(x):
    return 2**x * torch.sin(2**-x)


# Metric calculation
def metric(pred, target):
    return (pred-target).abs().mean()


# Train dataset
# Amount of points
# x_train = torch.rand(100)
x_train = torch.linspace(-10, 5, 100)
# Centering the graph
# x_train = x_train * 20.0 - 10.0
# y_train = torch.sin(x_train)
y_train = target_function(x_train)

# Adding noise and creating training sample
# noise = torch.randn(y_train.shape) / 5.
noise = torch.randn(y_train.shape) / 20.
y_train = y_train + noise


# Converting x_train and y_train vectors to columns where ones' row contains a value
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

# Validation dataset
# x_validation = torch.linspace(-10, 10, 100)
x_validation = torch.linspace(-10, 5, 100)
y_validation = target_function(x_validation)
plt.plot(x_validation.numpy(), y_validation.numpy(), 'o')
# plt.title('sin(x)')
plt.title('Target function')
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)


# Creating class for the neural network
# SineNet inherits from the class torch.nn.Module (base class for all neural network modules)
class SineNet(torch.nn.Module):
    # __init__ function for transferring the quantity of hidden neurons
    def __init__(self, n_hidden_neurons):
        super(SineNet, self).__init__()
        # Fully connected layer (the first layer)
        # Transferring the amount of input and output neurons
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
        # Activation function
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 1)

    # This function describes how the layers are sequentially applied
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


# Creating the neural network itself
# The number of hidden neurons is given as a parameter
sine_net = SineNet(200)


# Predict function
def predict(net, x, y):
    y_pred = net.forward(x)
    plt.plot(x.numpy(), y.numpy(), 'o', c='b', label='Ground truth')
    plt.plot(x.numpy(), y_pred.data.numpy(), 'o', c='r', label='Prediction')
    plt.legend(loc='upper left')
    plt.xlabel('$x$')
    plt.ylabel('$y$')


# Test prediction (w/o training)
# predict(sine_net, x_validation, y_validation)
# plt.show()

# Using optimizer for learning
optimizer = torch.optim.Adam(sine_net.parameters(), lr=0.8)


# Loss function
def loss(pred, target):
    squares = (pred - target) ** 2
    return squares.mean()


# Training
for epoch_index in range(1800):
    optimizer.zero_grad()
    y_pred = sine_net.forward(x_train)
    loss_val = loss(y_pred, y_train)
    loss_val.backward()
    optimizer.step()

predict(sine_net, x_validation, y_validation)
plt.show()

# Display metric value
print(metric(sine_net.forward(x_validation), y_validation).item())
