import numpy as np

#Setting layer length and number of nodes per layer
length = 5
nodes = [2, 3, 3, 3, 2]

#Initializing weights for each layer
W1 = np.random.randn(nodes[1], nodes[0])
W2 = np.random.randn(nodes[2], nodes[1])
W3 = np.random.randn(nodes[3], nodes[2])
W4 = np.random.randn(nodes[4], nodes[3])
W5 = np.random.randn(nodes[4], nodes[3])

#Initializing biases for each layer
B1 = np.random.randn(nodes[1], 1)
B2 = np.random.randn(nodes[2], 1)
B3 = np.random.randn(nodes[3], 1)
B4 = np.random.randn(nodes[4], 1)
B5 = np.random.randn(nodes[4], 1)

#Function to prepare the input data
#These values will later be taken from a CSV file
def prep_data():
    data = np.array([
        [150, 70],
        [254, 73],
        [312, 68],
        [120, 60],
        [154, 61],
        [212, 65],
        [216, 67],
        [145, 67],
        [184, 64],
        [130, 69]
        ])
    
    labels = np.array([0,1,1,0,0,1,1,0,1,0])
    m = len(data)

    #Transposing the input data to match the expected shape
    A0 = data.T
    labels = labels.reshape(nodes[length], m)

    return A0, labels, m

#Cost function using binary cross-entropy
def cost(y_hat, y):
    losses = -y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)

    #turning our output into a n^L x m vector
    m = y_hat.reshape(-1).shape[0]

    #Calculating the mean loss over all examples
    sum_loss = (1 / m) * np.sum(losses, axis=1)

    return np.sum(sum_loss)

#Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#Forward prop function
def feed_forward(A0):
    #Calculating the activations for each layer
    Z1 = np.dot(W1, A0) + B1
    A1 = sigmoid(Z1)

    Z2 = np.dot(W2, A1) + B2
    A2 = sigmoid(Z2)

    Z3 = np.dot(W3, A2) + B3
    A3 = sigmoid(Z3)

    Z4 = np.dot(W4, A3) + B4
    A4 = sigmoid(Z4)

    Z5 = np.dot(W5, A4) + B5
    A5 = sigmoid(Z5)

    #Storing the activations in a cache for backprop
    cache = {
        "A0": A0,
        "A1": A1,
        "A2": A2,
        "A3": A3,
        "A4": A4,
    }

    return A5, cache

A0, labels, m = prep_data()
