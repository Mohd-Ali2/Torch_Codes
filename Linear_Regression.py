# Importing Dependencies

import torch 
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Prepareing Data

x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=0)

# Casting Numpy to Tensor

x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1) # Reshaping into 2D

n_samples, n_features = x.shape

# Working on Model Linear

input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# Loss and Optimizer

learning_rate = 0.1
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training Loop 

num_epochs = 100

for epoch in range(num_epochs):
    # Forward Pass and loss
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # Backward Pass and Update

    loss.backward()
    optimizer.step()

    # Zero grad before new step 

    optimizer.zero_grad()

    if (epoch+1) % 10 ==0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.3f}')
    
# Ploting 

pred = model(x).detach().numpy()

plt.plot(x_numpy, y_numpy, 'ro')
plt.plot(x_numpy, pred, 'b')
plt.show()

