import numpy as np

#Setting layer length and number of nodes per layer
length = 3
nodes = [2, 3, 3, 1]

#Initializing weights for each layer
weights = []
for i in range(length):
    W = np.random.randn(nodes[i + 1], nodes[i])
    weights.append(W)

#Initializing biases for each layer
biases = []
for i in range(length):
    B = np.random.randn(nodes[i + 1], 1)
    biases.append(B)

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
            A = y_hat
            A_back = cache[i - 1]
            
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

    return error

def train():
    global weights, biases
    epochs = 100
    alpha = 0.1
    costs = []

    A0, labels, m = prep_data()

    for e in range(epochs + 1):
        error = backprop(length, A0, labels, weights, biases, alpha, m)
        costs.append(error)

        if e % 20 == 0:
            print(f"epoch {e}: cost = {error:4f}")

    return costs

train()