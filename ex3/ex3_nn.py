import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from functions import (
    display_data, predict
)

plt.ion()

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

# ########## Part1: Loading and Visualizing Data ##########
print('Loading and Visualizing Data ...\n')

data = sio.loadmat('ex3data1.mat')
X = data['X']
y = data['y']

m = X.shape[0]

rand_indices = np.random.permutation(m)
sel = X[rand_indices[:100], :]

display_data(sel)

input('Program paused. Press enter to continue.\n')

# ########## Part2: Loading Parameters ##########
print('Loading Saved Neural Network Parameters ...\n')
data = sio.loadmat('ex3weights.mat')
Theta1 = data['Theta1']
Theta2 = data['Theta2']

# ########## Part3: Implement Predict ##########
pred = predict(Theta1, Theta2, X)

print('Train Accuracy: {0:.1f}'.format(np.mean(pred == y) * 100))

input('Program paused. Press enter to continue.\n')

rp = np.random.permutation(m)

for i in range(m):
    print('Displaying Example Image\n')
    display_data(X[rp[i], :].reshape(1, -1))

    pred = predict(Theta1, Theta2, X[rp[i], :].reshape(1, -1))
    print(
        '\nNeural Network Prediction: {0} (digit {1})\n'
        .format(pred[0, 0], np.mod(pred, 10)[0, 0])
    )

    s = input('Paused - press enter to continue, q to exit:')
    if s == 'q':
        break

plt.close()
