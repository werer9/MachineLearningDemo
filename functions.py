import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd


def read_data(filename: str):
    """
    Read data from csv
    :param filename: Filename of csv
    :return: Numpy array of csv data
    """
    return pd.read_csv(filename, header=None).to_numpy()


def time_execution(func, *args):
    """
    Calculate the execution time of a function
    :param func: Function to measure execution time of
    :param args: Arguments to be passed to the function
    :return: Time it takes for the function to execute in us
    """
    t1 = time.perf_counter()
    func(*args)
    t2 = time.perf_counter()
    return (t2 - t1) * 10 ** 6


def normal_distribution(size: int):
    """
    Generate a range of normally distributed values
    :param size: Number of elements in array
    :return: Array containing normally distributed scalars
    """
    rng = np.random.default_rng()
    return rng.standard_normal(size)


def uniform_distribution(low: int, high: int, size: int):
    """
    Convert uniform distribution range to fit between a range of integers
    in a random order
    :param low: Lowest value
    :param high: Highest value
    :param size: Number of elements to have in array
    :return: Array of elements which values are uniformly distributed
    """
    return np.random.uniform(low, high, size)


def test_plot(func, title: str, distribution_func, args: [int], save_plot: bool = True):
    """
    Generate a plot of each function and its derivative function
    :param func: Function to plot
    :param title: Name of function for displaying on plot
    :param distribution_func: Function to use to generate distribution of
    values input into function
    :param args: Arguments for distribution function
    :param save_plot: Should the plot be saved as a .jpg? True/False.
    Default value is True
    :return: void
    """
    # generate distribution of random values
    x = distribution_func(*args)
    # Arrange values in order so gradient can be calculated
    x = np.sort(x)
    # f(x)
    f_x = func(x)
    # generate two plots next two each other
    fig, axs = plt.subplots(1, 2)
    # Create enough spacing to label axes of both plots
    fig.tight_layout(pad=3.0)
    axs[0].plot(x, f_x, '-')
    axs[0].set_title(title)
    axs[0].grid()
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("f(x)")
    # Use numpy numeric gradient to plot derivative
    axs[1].plot(x, np.gradient(f_x, x), '-')
    axs[1].set_title(title + " Derivative")
    axs[1].grid()
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("f'(x)")
    # If save_plot is True, save the plots as .jpg
    if save_plot:
        plt.savefig("plots/" + title + ".jpg")


def speed_test(name: str, func1, func2=None, sample_size: int = 10000,
               iterations: int = 5):
    """
    Test the speed of function and compare it with the speed of another
    function
    :param sample_size: Number of samples to test
    :param iterations: number of timings to take
    :param name: Name of the function
    :param func1: First function to test
    :param func2: Second function to test
    :return: void
    """
    times1 = []
    times2 = []
    # Test function n times
    for i in range(0, iterations):
        x = normal_distribution(sample_size)
    times1.append(time_execution(func1, [x]))
    # Test function 2 if one is supplied
    if func2 is not None:
        times2.append(time_execution(func2, [x]))
    print(name + " self-made function times: ")
    for i in times1:
        print(f"{i:.2f}us")
    print(f"Mean time: {np.mean(times1):.2f}us")
    if func2 is not None:
        print(name + " tensorflow function times: ")
    for i in times2:
        print(f"{i:.2f}us")
    print(f"Mean time: {np.mean(times2):.2f}us")


def sigmoid(x):
    """
    Apply sigmoid function
    :param x: Numeric value to input as function parameter
    :return: Output of function
    """
    expo = np.exp(-x)
    return 1 / (1 + expo)


def sigmoid_derivative(x):
    """
    Derivative of sigmoid function
    :param x: Numeric value to input as function parameter
    :return: Output of function
    """
    return sigmoid(x) * (1 - sigmoid(x))


def tansig(x):
    """
    Apply tansig function
    :param x: Numeric value to input as function parameter
    :return: Output of function
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tansig_derivative(x):
    """
    Derivative of tansig function
    :param x: Numeric value to input as function parameter
    :return: Output of function
    """
    return 1 - tansig(x) ** 2


def eliotsig(x):
    """
    Apply eliotsig function
    :param x: Numeric value to input as function parameter
    :return: Output of function
    """
    return x / (1 + np.absolute(x))


def softplus(x):
    """
    Apply softplus function
    :param x: Numeric value to input as function parameter
    :return: Output of function
    """
    return np.log(1 + np.exp(x))


def elu(x, alpha: float = 1.):
    """
    Apply ELU function
    :param x: Numeric value to input as function parameter
    :param alpha: alpha co-efficient
    :return: Output of function
    """
    x = np.where(x < 0, alpha * (np.exp(x) - 1), x)
    return x


def soft_plus_plus(x, k: float = 1., c: float = 2.):
    """
    Apply soft++ function
    :param c: c co-efficient
    :param k: k co-efficient
    :param x: Numeric value to input as function parameter
    :return: Output of function
    """
    return np.log(1 + np.exp(k * x)) + x / c - np.log(2)


def sqnl(x):
    """
    Apply SQNL function
    :param x: Numeric value to input as function parameter
    :return: Output of function
    """
    # make a copy of the actual values, not a pointer.
    original = np.copy(x)
    # Find all values less the -2 and replace them with -1
    x = np.where(original < -2., -np.ones_like(x), x)
    # Find all values that were originally -2 <= x <= 0 and pass them
    # through equation
    x = np.where(np.logical_and(original >= -2., original <= 0.), original
                 + np.power(original, 2) / 4, x)
    # Find all values that were originally 0 < x <= 2 and pass them through
    # equation
    x = np.where(np.logical_and(original > 0., original <= 2.), original -
                 np.power(original, 2) / 4, x)
    # Find all values that were originally greater than two and replace
    # them with ones
    x = np.where(original > 2., np.ones_like(x), x)
    return x


def cost(y: np.array, yp: np.array):
    return 0.5 * (y - yp) ** 2


def mse(y: np.array, yp: np.array):
    return ((y - yp) ** 2).mean()
