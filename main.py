import numpy as np
import csv
import matplotlib.pyplot as plt

#Setting layer length and number of nodes per layer
length = 3
nodes = [30, 12, 12, 1]

#Initialising weights for each layer
weights = []
for i in range(length):
    W = np.random.randn(nodes[i + 1], nodes[i])
    weights.append(W)

#Initialising biases for each layer
biases = []
for i in range(length):
    B = np.random.randn(nodes[i + 1], 1)
    biases.append(B)

#Function to prepare the input data
def prep_data(seed=1, ratio=0.3):
    with open("data.csv", "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row

        raw_data = [row for row in csv_reader]

    #Setting the random seed for reproduction
    np.random.seed(seed)

    #Splitting the data into labels and features
    labels = np.array([1 if row[1] == 'M' else 0 for row in raw_data])
    data = np.array([list(map(float, row[2:])) for row in raw_data])

    # Shuffling the data
    shuffle = np.arange(len(data))
    np.random.shuffle(shuffle)

    data = data[shuffle]
    labels = labels[shuffle]

    #Normalising the data
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data - mean) / std

    #Splitting the data into training and testing sets
    split_index = int((1 - ratio) * len(data))
    train_data = data[:split_index]
    train_labels = labels[:split_index]
    test_data = data[split_index:]
    test_labels = labels[split_index:]

    #Transposing and reshaping for training
    A0_train = train_data.T
    A0_test = test_data.T
    train_labels = train_labels.reshape(1, -1)
    test_labels = test_labels.reshape(1, -1)

    m = len(A0_train[0])

    return A0_train, train_labels, A0_test, test_labels, m

#Cost function using binary cross-entropy
def cost(y_hat, y):
    losses = -y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)

    #turning the output into a n^L x m vector
    m = y_hat.reshape(-1).shape[0]

    #Calculating the mean loss over all examples
    sum_loss = (1 / m) * np.sum(losses, axis=1)

    return np.sum(sum_loss)

#Sigmoid activation function
def sigmoid(z):
    #Prevent overflow
    z = np.clip(z, -500, 500)  
    return 1 / (1 + np.exp(-z))

#Forward prop function
def feed_forward(A0, length):
    cache = [A0]
    #Calculating the activations for each layer
    for i in range(length):
        if i != length - 1:
            Z = np.dot(weights[i], cache[i]) + biases[i]
            A = sigmoid(Z)
            #Storing the activations in a cache for backprop
            cache.append(A)

        else:
            Z = np.dot(weights[i], cache[i]) + biases[i]
            A_last = sigmoid(Z)

    return A_last, cache

#Backprop function with scalability in mind
def backprop(length, A0, labels, weights, biases, alpha, m):
    y_hat, cache = feed_forward(A0, length)

    error = cost(y_hat, labels)

    for i in range(length, 0, -1):
        if i == length:
            #For the output layer, we do not apply the sigmoid function
            A = y_hat
            A_back = cache[i - 1]
            
            #Calculating the gradient of the cost with respect to the activation of this layer
            dCost_dOut = (1 / m) * (A - labels)
            assert dCost_dOut.shape == (nodes[i], m)

            #Calculating the gradient of the output layer with respect to the activation of the previous layer
            dOut_dWeight = A_back
            assert dOut_dWeight.shape == (nodes[i - 1], m)

            #Calculating the gradient of the cost with respect to the weights of this layer
            dCost_dWeight = np.dot(dCost_dOut, dOut_dWeight.T)
            assert dCost_dWeight.shape == (nodes[i], nodes[i - 1])

            #Calculating the gradient of the cost with respect to the biases of this layer
            dCost_dBias = np.sum(dCost_dOut, axis=1, keepdims=True)
            assert dCost_dBias.shape == (nodes[i], 1)

            #Calculating the gradient of the cost with respect to the activation of previous layer
            dCost_dA_back = weights[i - 1].T @ dCost_dOut
            assert dCost_dA_back.shape == (nodes[i - 1], m)

            #Updating the weights and biases
            weights[i - 1] -= alpha * dCost_dWeight
            biases[i - 1] -= alpha * dCost_dBias

        else:
            A = cache[i]

            if i == 1:
                A_back = A0

            else:
                A_back = cache[i - 1]

            #Calculating the gradient of the cost with respect to the activation of this layer
            dA_dCost = A * (1 - A)
            dSigmoid = dCost_dA_back * dA_dCost
            assert dSigmoid.shape == (nodes[i], m)

            #Calculating the gradient of the output layer with respect to the activation of the previous layer
            dOut_dWeight = cache[i - 1]
            dCost_dWeight = np.dot(dSigmoid, dOut_dWeight.T)
            assert dCost_dWeight.shape == (nodes[i], nodes[i - 1])

            #Calculating the gradient of the cost with respect to the biases of this layer
            dCost_dBias = np.sum(dSigmoid, axis=1, keepdims=True)
            assert dCost_dBias.shape == (nodes[i], 1)

            dCost_dA_back = weights[i - 1].T @ dSigmoid
            assert dCost_dA_back.shape == (nodes[i - 1], m)

            #Updating the weights and biases
            weights[i - 1] -= alpha * dCost_dWeight
            biases[i - 1] -= alpha * dCost_dBias
        
        #Calculating the error and accuracy for the current epoch
        y_hat, _ = feed_forward(A0, length)
        predictions = (y_hat > 0.5).astype(int)
        accuracy = np.mean(predictions == labels)

    return [error, accuracy]

def train():
    global weights, biases
    epochs = 500
    alpha = 0.1
    costs = []
    accuracies = []

    A0_train, labels_train, A0_test, labels_test, m = prep_data()

    for e in range(epochs + 1):
        error = backprop(length, A0_train, labels_train, weights, biases, alpha, m)
        costs.append(error[0])
        accuracies.append(error[1])

        if e % 10 == 0:
            print("Epoch ", e, ": cost =", round(error[0], 4))

    #Final evaluation on the test set
    y_hat, _ = feed_forward(A0_test, length)
    predictions = (y_hat > 0.5).astype(int)
    accuracy = np.mean(predictions == labels_test)

    print("Accuracy: ", round(accuracy*100.0, 2), "%")

    #Plotting cost and accuracy on a graph over epochs
    plt.plot(range(epochs + 1), costs, label="Cost")
    plt.plot(range(epochs + 1), accuracies, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Cost and Accuracy over Epochs")
    plt.grid(True)
    plt.show()

train()