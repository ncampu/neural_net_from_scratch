# Dependencies
import pandas as pd
import numpy as np
import pickle

# Load the dataset from a CSV file and convert it into a NumPy array
data = pd.read_csv(r'C:/Users/Nico/Python_Projects/Ayalytical/CuDDI/CNN/digit-recognizer/train.csv')
data = np.array(data)

num_samples, num_pixels = data.shape  # Get the number of samples and pixels (including the label as the first column)
np.random.shuffle(data) # shuffle before splitting into training and validation sets

data_val = data[0:1000].T  # Use first 1000 samples as validation set
y_val = data_val[0]  # Labels
x_val = data_val[1:num_pixels]  # Features
x_val = x_val / 255.  # Normalize the pixel values to be between 0 and 1

data_train = data[1000:num_samples].T  # Use remaining data as training set
y_train = data_train[0]
x_train = data_train[1:num_pixels]
x_train = x_train / 255.


def init_params():
    # Initialize weights and biases for the 3 layers
    w1 = np.random.rand(200, 784) - 0.5  # Weights for 784 input pixels (0th layer)
    b1 = np.random.rand(200, 1) - 0.  # Biases for 200 neurons (1st hidden layer)
    w2 = np.random.rand(50, 200) - 0.5  # Weights for 50 neurons (2nd hidden layer)
    b2 = np.random.rand(50, 1) - 0.5  # Biases for 50 neurons (2nd hidden layer)
    w3 = np.random.rand(10, 50) - 0.5  # Weights for 10 neurons (3rd hidden layer)
    b3 = np.random.rand(10, 1) - 0.5  # Biases for 10 neurons (3rd hidden layer)
    return w1, b1, w2, b2, w3, b3


def relu(Z):
    # ReLU activation function, where relu(x) = x if x > 0, and relu(x) = 0 if x <= 0
    return np.maximum(Z, 0)  # Return maximum between Z and 0


def softmax(Z):
    # Softmax activation function for multiclass classification
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_propagation(w1, b1, w2, b2, w3, b3, x):
    # Perform forward propagation through the network
    Z1 = w1.dot(x) + b1  # Linear transformation for layer 1
    A1 = relu(Z1)  # Apply ReLU activation for layer 1
    Z2 = w2.dot(A1) + b2  # Linear transformation for layer 2
    A2 = relu(Z2)  # Apply ReLU activation for layer 2
    Z3 = w3.dot(A2) + b3  # Linear transformation for layer 3
    A3 = softmax(Z3)  # Apply softmax activation for the output layer
    return Z1, A1, Z2, A2, Z3, A3


def relu_deriv(Z):
    # Derivative of ReLU function, returns 1 for positive Z and 0 otherwise
    return Z > 0


def one_hot(y):
    # One-hot encode the output labels
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def backward_propagation(Z1, A1, Z2, A2, Z3, A3, w1, w2, w3, x, y):
    # Perform backward propagation to calculate gradients (how much we should change our weights and biases)
    one_hot_y = one_hot(y)
    dZ3 = A3 - one_hot_y  # Calcualte output errors
    dw3 = 1 / num_samples * dZ3.dot(A2.T)  # Gradient of weights for layer 3
    db3 = 1 / num_samples * np.sum(dZ3, axis=1).reshape(-1, 1)  # Gradient of biases for layer 3
    dZ2 = w3.T.dot(dZ3) * relu_deriv(Z2)  # Calculate error of layer 2
    dw2 = 1 / num_samples * dZ2.dot(A1.T)  # Gradient of weights for layer 2
    db2 = 1 / num_samples * np.sum(dZ2, axis=1).reshape(-1, 1)  # Gradient of biases for layer 2
    dZ1 = w2.T.dot(dZ2) * relu_deriv(Z1)  # Calculate error of layer 2
    dw1 = 1 / num_samples * dZ1.dot(x.T)  # Gradient of weights for layer 1
    db1 = 1 / num_samples * np.sum(dZ1, axis=1).reshape(-1, 1)  # Gradient of biases for layer 2
    return dw1, db1, dw2, db2, dw3, db3


def update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha):
    # Update the weights and biases using the calculated gradients and learning rate
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    w3 = w3 - alpha * dw3
    b3 = b3 - alpha * db3
    return w1, b1, w2, b2, w3, b3


def get_predictions(A2):
    # Get the predicted labels by finding the index of the maximum value in the output layer
    return np.argmax(A2, 0)


def get_accuracy(predictions, y):
    # Calculate the accuracy of predictions by comparing with true labels
    print(predictions, y)
    return np.sum(predictions == y) / y.size

def gradient_descent(x, y, alpha, iterations):
    # Train the model using gradient descent
    w1, b1, w2, b2, w3, b3 = init_params()
    for i in range(iterations):
        # Perform forward propagation and calculate gradients
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(w1, b1, w2, b2, w3, b3, x)
        dw1, db1, dw2, db2, dw3, db3 = backward_propagation(Z1, A1, Z2, A2, Z3, A3, w1, w2, w3, x, y)
        # Update parameters with calculated gradients
        w1, b1, w2, b2, w3, b3 = update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha)
        if i % 10 == 0:  # Print progress every 10 iterations
            print("Iteration: ", i)
            predictions = get_predictions(A3)
            print(get_accuracy(predictions, y))
    return w1, b1, w2, b2, w3, b3  # Return the updated parameters


def save_model(filename, w1, b1, w2, b2, w3, b3):
    # Save the trained model parameters to a file using pickle
    with open(filename, 'wb') as file:
        pickle.dump((w1, b1, w2, b2, w3, b3), file)


# Train the model and save the parameters
w1, b1, w2, b2, w3, b3 = gradient_descent(x_train, y_train, 0.10, 1000)
save_model("my_custom_neural_network.pkl", w1, b1, w2, b2, w3, b3)


#validation_predictions = make_predictions(x_val, w1, b1, w2, b2)
#get_accuracy(validation_predictions, y_val)