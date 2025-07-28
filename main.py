import numpy as np

#Setting layer length and number of nodes per layer
length = 5
nodes = [2, 3, 3, 3, 2]

#Initializing weights for each layer
w1 = np.random.randn(nodes[1], nodes[0])
w2 = np.random.randn(nodes[2], nodes[1])
w3 = np.random.randn(nodes[3], nodes[2])
w4 = np.random.randn(nodes[4], nodes[3])

#Initializing biases for each layer
b1 = np.random.randn(nodes[1], 1)
b2 = np.random.randn(nodes[2], 1)
b3 = np.random.randn(nodes[3], 1)
b4 = np.random.randn(nodes[4], 1)

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

#Our sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))