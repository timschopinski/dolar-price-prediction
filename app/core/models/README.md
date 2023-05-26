

# FeedForward


The neural network used in the provided code is a feedforward neural network, specifically a multi-layer perceptron (MLP). Here's a breakdown of the neural network architecture:

The input layer has the same number of neurons as the number of features in the data.
There are two hidden layers with 32 and 16 neurons, respectively. These layers use the ReLU activation function, which introduces non-linearity to the model.
The output layer consists of a single neuron, as the goal is to predict a single continuous value (USD/PLN price). There is no activation function applied to the output layer, making it a linear layer.
The architecture of this neural network is sequential, meaning that the layers are stacked on top of each other sequentially. Each layer is fully connected to the next layer, meaning that all the neurons in a layer are connected to all the neurons in the subsequent layer.

The model is trained using the mean squared error (MSE) loss function and the Adam optimizer. The MSE loss function is commonly used for regression tasks, aiming to minimize the squared difference between the predicted and actual values.

Overall, this neural network architecture can be considered a basic MLP model suitable for regression problems like predicting USD/PLN prices.

