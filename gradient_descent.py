import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('TkAgg')


def polynomial_function(x):
    """
    Polynomial function for gradient descent
    :param x: x input
    :return: f(x)
    """
    return x ** 4 - (1 / 2) * x ** 2 + 1


def polynomial_derivative(x):
    """
    Derivative of polynomial function
    :param x: x input
    :return: f'(x)
    """
    return (4 * (x ** 3)) - x


def plot_polynomial(function):
    """
    Plot polynomial function
    :param function: Polynomial function to plot
    :return: None
    """
    x = np.linspace(-1, 1, 1000)
    fx = function(x)
    plt.plot(x, fx)
    plt.grid()
    plt.title("Polynomial Function Plot")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()


def gradient_descent(x_bar, eta, derivative):
    """
    Gradient descent algorithm
    :param x_bar: X value to start at
    :param eta: Descent rate
    :param derivative: Derivative of function being solved
    :return: Tuple with final x value and history of x values
    """
    g = derivative(x_bar)
    x_bar_history = [x_bar]
    while not -0.005 < g < 0.005:
        x_bar = x_bar - eta * g
        g = derivative(x_bar)
        # Add x update x value to history, so it can be plotted later
        x_bar_history.append(x_bar)

    return [x_bar, np.array(x_bar_history)]


def plot_gradient_descent(function, x_bar, eta, derivative):
    """
    Plot gradient descent and polynomial function
    :param function: Polynomial function to plot
    :param x_bar: X value to start at
    :param eta: Descent rate
    :param derivative: Derivative of function being solved
    :return: None
    """
    x = np.linspace(-1, 1, 1000)
    fx = function(x)
    plt.plot(x, fx)
    # Get x history from gradient descent
    [_, descent] = gradient_descent(x_bar, eta, derivative)
    plt.plot(descent, function(descent), 'rx-')
    plt.grid()
    plt.title("Gradient Descent Plot")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()


def max_learning_rate(x, eta, eta_rate, derivative):
    """
    Discover learning rate that changes the minimum from the gradient
    descent algorithm
    :param x: Starting x value for gradient descent
    :param eta: Learning rate
    :param eta_rate: Rate at which learning rate is incremented
    :param derivative: Derivative of function being solved
    :return:
    """
    [x_start, _] = gradient_descent(x, eta, derivative)
    x_bar = x_start
    # While the final x value is the same keep incrementing the learning
    # rate
    while x_start - 0.1 < x_bar < x_start + 0.1:
        try:
            [x_bar, _] = gradient_descent(x, eta, derivative)
        except OverflowError:
            # If the x value gets too large
            print("Overflow error")
            return eta
        eta += eta_rate
    return eta
