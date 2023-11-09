from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from functions import *


class ANN(object):

    def __init__(self, ni, nh, no, activation_function=sigmoid,
                 derivative_function=sigmoid_derivative):
        """
        ANN object constructor
        :param ni: Number of inputs
        :param nh: Number of neurons in hidden layer
        :param no: Number of outputs
        :param activation_function: Activation function to use
        :param derivative_function: Derivative of activation function
        """
        self.ni = ni
        self.nh = nh
        self.no = no
        self.w0 = np.random.rand(self.nh, self.ni)
        self.w1 = np.random.rand(self.no, self.nh)
        self.b0 = np.random.rand(self.nh)
        self.b1 = np.random.rand(self.no)
        self.h0 = np.zeros(self.nh)
        self.o = np.zeros(self.no)
        self.mse = []
        self.error = []
        self.activation_function = activation_function
        self.derivative_function = derivative_function

    def delta(self, y: np.array, yp: np.array, x):
        """
        Calculate the delta for the output layer
        :param self:
        :param y: Actual output
        :param yp: Predicted output
        :param x: Activation value passed into the activation function
        :return: Delta for output layer
        """
        return (y - yp) * self.derivative_function(x)

    def training_iteration(self, training_input, training_output, eta):
        """
        Runs for every training input
        :param self:
        :param training_input: Input values to propagate
        :param training_output: Correct output values
        :param eta: Learning rate
        :return: None
        """
        # Forward pass to generate values for each layer
        self.fwd_pass(training_input)
        # Calculating the error for the output layer
        delta_o = self.delta(training_output, self.o, (self.w1 @ self.h0 +
                                                       self.b1))
        # Updating the weight of the output layer
        self.w1 = self.w1 + eta * np.array([delta_o]).transpose() * self.h0
        # Update the bias
        self.b1 = self.b1 + eta * delta_o
        # Calculate the delta for the hidden layer
        delta_h = np.sum(delta_o @ self.w1) * self.derivative_function(self.w0 @ training_input + self.b0)
        # Update the hidden layer weights
        self.w0 = self.w0 + eta * np.array([delta_h]).transpose() * training_input
        # Update the hidden layer biases
        self.b0 = self.b0 + eta * delta_h

    def calculate_error(self, inputs, outputs):
        """
        Calculate the error for each input and output
        :param self:
        :param inputs: ANN input
        :param outputs: Correct output
        :return: None
        """
        predictions = []
        # Run test input through network
        for test_input in inputs:
            [_, output] = self.fwd_pass(test_input)
            predictions.append(output)
        predictions = np.array(predictions)
        # Quantify error between predictions and ground truth and add to
        # array with all error values calculated
        self.mse.append(mse(outputs, predictions))
        self.error.append(np.mean(cost(outputs, predictions), axis=0))

    def train(self, training_inputs, training_outputs, epochs: int, eta: float,
              testing_inputs: np.array = None, testing_outputs: np.array = None):
        """
        Train the neural network
        :param self:
        :param training_inputs: Inputs to use for training
        :param training_outputs: Outputs to use for training
        :param epochs: Number of times the function iterates over training
        data
        :param eta: Learning rate
        :param testing_inputs: Inputs that can be used for testing
        :param testing_outputs: Outputs that can be used for testing
        :return: None
        """
        # Reset all ANN values
        epoch = 0
        self.w0 = np.random.rand(self.nh, self.ni)
        self.w1 = np.random.rand(self.no, self.nh)
        self.b0 = np.random.rand(self.nh)
        self.b1 = np.random.rand(self.no)
        self.h0 = np.zeros(self.nh)
        self.o = np.zeros(self.no)
        self.mse = []
        self.error = []
        # Until the number of epochs is reach keep training
        while epoch < epochs:
            # Run training through all training data
            for _, (training_input, training_output) in enumerate(zip(training_inputs, training_outputs)):
                self.training_iteration(training_input, training_output,
                                        eta)
            # Test ANN on test data if available otherwise test on training
            # data
            if testing_outputs is None or testing_inputs is None:
                self.calculate_error(training_inputs, training_outputs)
            else:
                self.calculate_error(testing_inputs, testing_outputs)
                epoch += 1

    def plot_training_mse(self, title: str = ""):
        """
        Plot the mse during the training
        :param self:
        :param title: Title for graph
        :return: None
        """
        plt.plot(self.mse)
        plt.title("Training Mean Squared Error " + title)
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.grid()
        plt.show()

    def compare_learning_rates(self, training_inputs, training_outputs,
                               epochs: int, eta: float,
                               testing_inputs: np.array = None,
                               testing_outputs: np.array = None, title: str
                               = ""):
        """
        try different learning rates
        :param self:
        :param training_inputs: Inputs to use for training
        :param training_outputs: Outputs to use for training
        :param epochs: Number of times the function iterates over training
        data
        :param eta: Learning rate
        :param testing_inputs: Inputs that can be used for testing
        :param testing_outputs: Outputs that can be used for testing
        :param title: Title for the graph
        :return: None
        """
        # Try different training rate values in increments of 0.1 for 5
        # iterations and then plot
        for i in range(0, 5):
            self.train(training_inputs, training_outputs, epochs, eta,
                       testing_inputs, testing_outputs)
            self.plot_training_mse(title=title + f" {eta:.1f}")
            eta += 0.1

    def fwd_pass(self, i):
        """
        Pass data through ANN
        :param self:
        :param i: Input data
        :return: Tuple of arrays - hidden layer and output layer
        """
        self.b0 = self.b0.flatten()
        self.b1 = self.b1.flatten()
        # @ symbol represents matrix multiplication in numpy
        self.h0 = self.activation_function(self.w0 @ i + self.b0)
        self.o = self.activation_function(self.w1 @ self.h0 + self.b1)
        return [self.h0, self.o]


def load_iris_training_data() -> [np.array, np.array, np.array, np.array]:
    """
    Load iris test dataset
    :return: Training and testing sets
    """
    df = pd.read_csv("Iris.csv")
    # Remove index column
    df = df.drop('Id', axis=1)
    # One hot encode species column
    y = pd.get_dummies(df['Species'])
    # Remove species column
    df = df.drop('Species', axis=1)
    # Normalize x values
    x = normalize(df.to_numpy())
    y = y.to_numpy()
    # Return values split into training and testing data
    return train_test_split(x, y, test_size=0.2)


def generate_xor_data() -> [np.array, np.array, np.array, np.array]:
    """
    Generate XOR training dataset
    :return: Training and testing sets
    """
    # Generate random values between -1 and 1
    x = (np.random.rand(500, 2) - 0.5) * 2
    # Generate y values
    txor = np.ones(500) * -1
    txor = np.where(txor, np.logical_and(x[:, 0] > 0, x[:, 1] < 0), 1)
    txor = np.where(txor, np.logical_and(x[:, 0] < 0, x[:, 1] > 0), 1)
    y = np.stack((txor, txor * -1)).T
    # Return values split into training and testing data
    return train_test_split(x, y, test_size=0.2)
