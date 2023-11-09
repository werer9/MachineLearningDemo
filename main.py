from gradient_descent import *
from ann import *

if __name__ == "__main__":
    # Get max learning rate
    print(f"Max learning rate: {max_learning_rate(0.85, 0.1, 0.1, polynomial_derivative)}")
    # Plot gradient descent and polynomial
    plot_gradient_descent(polynomial_function, 0.85, .1, polynomial_derivative)
    # Get training data and train newtork
    [training_inputs, testing_inputs, training_outputs, testing_outputs] = load_iris_training_data()
    ann = ANN(4, 10, 3)
    ann.train(training_inputs, training_outputs, epochs=1000, eta=.2,
              testing_inputs=testing_inputs,
              testing_outputs=testing_outputs)
    ann.plot_training_mse("Iris")
    # Compare learning rates
    ann.compare_learning_rates(training_inputs, training_outputs,
                               epochs=100, eta=.2, testing_inputs=testing_inputs,
                               testing_outputs=testing_outputs,
                               title="Iris")
    [training_inputs, testing_inputs, training_outputs, testing_outputs] = generate_xor_data()
    ann = ANN(2, 4, 2, activation_function=tansig,
              derivative_function=tansig_derivative)
    ann.train(training_inputs, training_outputs, epochs=200, eta=.2,
              testing_inputs=testing_inputs,
              testing_outputs=testing_outputs)
    ann.plot_training_mse("XOR")
    ann.compare_learning_rates(training_inputs, training_outputs,
                               epochs=100, eta=.2, testing_inputs=testing_inputs,
                               testing_outputs=testing_outputs,
                               title="XOR")
