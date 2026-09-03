# ===============================================================================
# DLFBT 2026/2027
# Lab assignment 1
# Authors:
#   Name1 NIA1 (complete your name and NIA here)
#   Name2 NIA2 (complete your name and NIA here)
# ===============================================================================

import numpy as np
import tensorflow as tf


class LinearRegressionModel(object):
    """
    Linear regression model for exercise 1

    Parameters
    ----------
    d : int
        Dimension

    Attributes
    ----------
    w : array
        Weight vector of shape (d, 1)
    b : array
        Bias term of shape (1, 1)
    """

    def __init__(self, d=2):
        self.w = np.random.randn(d, 1)
        self.b = np.random.randn(1, 1)

    def predict(self, x):
        """
        Predicts output y for input batch x

        Parameters
        ----------
        x : array
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension

        Returns
        -------
        y : array
            Ouput batch of shape (N, 1) which is the result of applying the
            linear regression model to the input x
        """

        # --- TO-DO block: Compute the model output y
        pass
        # --- End of TO-DO block

        return y

    def compute_gradients(self, x, t):
        """
        Calculates the gradients of the loss function with respect to the model
        parameters b and w, for an input batch (x, t)

        Parameters
        ----------
        x : array
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (N, 1) with the target values for each x

        Returns
        -------
        db : array
             Gradient of the loss with respect to the bias, shape (1, 1)
        dw : array
             Gradient of the loss with respect to the weights, shape (d, 1)
        """
        y = self.predict(x)

        # --- TO-DO block: Compute the gradients db and dw
        pass
        # --- End of TO-DO block

        return db, dw

    def gradient_step(self, x, t, eta):
        """
        Updates the model parameters with an input batch (x, t)

        Parameters
        ----------
        x : array
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (N, 1) with the target values for each x
        eta : float
            Learning rate
        """
        db, dw = self.compute_gradients(x, t)

        # --- TO-DO block: Update the model parameters b and w
        pass
        # --- End of TO-DO block

    def fit(self, x, t, eta, num_iters):
        """
        Trains the model for a fixed number of iterations

        Parameters
        ----------
        x : array
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (N, 1) with the target values for each x
        eta : float
            Learning rate
        num_iters : int
            Number of training iterations

        Returns
        -------
        loss : array
             Array of shape (num_iters,) with the loss after each training
             iteration
        """
        loss = np.zeros(num_iters)
        for i in range(num_iters):
            self.gradient_step(x, t, eta)
            loss[i] = self.get_loss(x, t)
        return loss

    def get_loss(self, x, t):
        """
        Calculates the MSE loss for an input batch (x, t)

        Parameters
        ----------
        x : array
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (N, 1) with the target values for each x

        Returns
        -------
        loss : float
             MSE loss
        """
        y = self.predict(x)
        loss = 0.5 * np.mean((y - t) ** 2)
        return loss


class LogisticRegressionModel(LinearRegressionModel):
    """
    Logistic regression model for exercise 2

    Parameters
    ----------
    d : int
        Dimension

    Attributes
    ----------
    w : array
        Weight vector of shape (d, 1)
    b : array
        Bias term of shape (1, 1)
    """

    def __init__(self, d=2):
        LinearRegressionModel.__init__(self, d)

    def sigmoid(z):
        """
        Calculates the sigmoid function on input z, element-wise

        Parameters
        ----------
        z : array or float
            Input value or values, of arbitrary shape

        Returns
        -------
        _ : array or float
            Sigmoid function evaluated on z, same shape as z
        """
        return 1.0 / (1.0 + np.exp(-z))

    # --- TO-DO block: Overwrite the methods of the LinearRegressionModel class
    pass
    # --- End of TO-DO block

    def get_loss(self, x, t):
        """
        Calculates the cross-entropy loss for an input batch (x, t)

        Parameters
        ----------
        x : array
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (N, 1) with the target labels for each x

        Returns
        -------
        loss : float
             Cross-entropy loss
        """
        y = self.predict(x)
        loss = -np.mean(t * np.log(y) + (1.0 - t) * np.log(1.0 - y))
        return loss


class BasicTF:
    """
    Static methods for exercises 3 and 4
    """

    @staticmethod
    def differentiate(f, x):
        """
        Calculates the derivative of the funcion f for each point in x

        Parameters
        ----------
        f : function
            A function of one variable with auto-differentiation, such as tf.cos
            or tf.exp
        x : array
            Input array of arbitrary shape with the points where the derivative of
            the function f must be evaluated

        Returns
        -------
        dy_dx : array
            Array with the same shape as x with the derivative
        """
        x = tf.Variable(x)

        # --- TO-DO block: Define the computational graph within a gradient tape and
        # --- compute the gradient
        pass
        # --- End of TO-DO block

        return dy_dx

    @staticmethod
    def gradient_descent(f, x0, eta, niters):
        """
        Minimizes a function using gradient descent

        Parameters
        ----------
        f : function
            A function of one variable x
        x : float
            The initial value of x
        eta : float
            The learning rate
        niters : int
            The number of iterations

        Returns
        -------
        xvals : array
            An array with shape (niters,) with the value of x after each iteration
        """
        x = tf.Variable(x0)

        x_history = []
        for i in range(niters):
            # --- TO-DO block: Define the computational graph within a gradient tape and
            # --- compute the gradient
            pass
            # --- End of TO-DO block

            # --- TO-DO block: Update the value of x using the tf.Variable assign method
            pass
            # --- End of TO-DO block

            x_history.append(x.numpy())

        xvals = np.array(x_history)

        return xvals


class LinearRegressionModel_TF(object):
    """
    Linear regression model using TensorFlow for exercise 5

    Parameters
    ----------
    d : int
        Dimension

    Attributes
    ----------
    w : tf.Variable
        Weight vector of shape (d, 1)
    b : tf.Variable
        Bias term of shape (1, 1)
    """

    def __init__(self, d=2):
        self.w = tf.Variable(tf.random.normal(shape=[d, 1], dtype=tf.dtypes.float64))
        self.b = tf.Variable(tf.random.normal(shape=[1, 1], dtype=tf.dtypes.float64))

    def predict(self, x):
        """
        Predicts output y for input batch x

        Parameters
        ----------
        x : array or tensor
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension

        Returns
        -------
        y : tensor
            Ouput batch of shape (N, 1) which is the result of applying the
            linear regression model to the input x

        """
        # --- TO-DO block: Compute the model output y
        pass
        # --- End of TO-DO block

        return y

    def compute_gradients(self, x, t):
        """
        Calculates the gradients of the loss function with respect to the model
        parameters b and w, for an input batch (x, t)

        Parameters
        ----------
        x : array or tensor
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension
        t : array or tensor
            Array of shape (N, 1) with the target values for each x

        Returns
        -------
        db : tensor
             Gradient of the loss with respect to the bias, shape (1, 1)
        dw : tensor
             Gradient of the loss with respect to the weights, shape (d, 1)
        """
        # --- TO-DO block: Compute the gradients db and dw of the loss function
        pass
        # --- End of TO-DO block

        return db, dw

    def gradient_step(self, x, t, eta):
        """
        Updates the model parameters with an input batch (x, t)

        Parameters
        ----------
        x : array or tensor
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension
        t : array or tensor
            Array of shape (N, 1) with the target values for each x
        eta : float
            Learning rate
        """
        db, dw = self.compute_gradients(x, t)

        # --- TO-DO block: Update the model parameters b and w
        pass
        # --- End of TO-DO block

    def fit(self, x, t, eta, num_iters):
        """
        Trains the model for a fixed number of iterations

        Parameters
        ----------
        x : array or tensor
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension
        t : array or tensor
            Array of shape (N, 1) with the target values for each x
        eta : float
            Learning rate
        num_iters : int
            Number of training iterations

        Returns
        -------
        loss : array
             Array of shape (num_iters,) with the loss after each training
             iteration
        """
        loss = np.zeros(num_iters)
        for i in range(num_iters):
            self.gradient_step(x, t, eta)
            loss[i] = self.get_loss(x, t).numpy()
        return loss

    def get_loss(self, x, t):
        """
        Calculates the MSE loss for an input batch (x, t)

        Parameters
        ----------
        x : array or tensor
            Input batch of shape (N, d), where N is the number of patterns and
            d is the dimension
        t : array or tensor
            Array of shape (N, 1) with the target values for each x

        Returns
        -------
        loss : float
             MSE loss
        """
        y = self.predict(x)
        loss = tf.reduce_mean(0.5 * (y - t) * (y - t))
        return loss


class NeuralNetwork(object):
    """
    Dense feedforward neural network model for exercise 6

    Parameters
    ----------
    d : list of tuples
        List of tuples that define the neural network architecture. There is one
        tuple of the form (n, a) for each layer, where n is the number of layer
        neurons and a is the type of layer. The following types are allowed:

        - 'sigmoid': a layer of sigmoid units
        - 'linear': a layer of linear units
        - 'input': a special layer for the input

        The minimum is two layers (input and output).
        The first layer must be always of type 'input'.
        Only one 'input' layer must be present, and it must be the first layer.
        The default value is layers=[(2, 'input'), (1, 'sigmoid')].
        The output layer must have only one neuron.

    Attributes
    ----------
    nlayers : int
        Number of layers, excluding the input layer
    W : list of arrays
        List containing the weight matrix for each processing layer
    b : list of arrays
        List containing the bias vector for each processing layer
    a : list of functions
        List containing the activation function for each processing layer
    da : list of functions
        List containing the derivative of the activation function for each
        processing layer
    """

    def __init__(self, layers=[(2, "input"), (1, "sigmoid")]):
        self.nlayers = len(layers) - 1
        self.W = []
        self.b = []
        self.a = []
        self.da = []

        # Set the weights, biases and activation functions:
        for l0, l1 in zip(layers[:-1], layers[1:]):
            self.W.append(np.random.randn(l1[0], l0[0]))
            self.b.append(np.random.randn(l1[0], 1))
            if l1[1] == "sigmoid":
                self.a.append(NeuralNetwork.sigmoid)
                self.da.append(NeuralNetwork.dsigmoid)
            elif l1[1] == "linear":
                self.a.append(NeuralNetwork.identity)
                self.da.append(NeuralNetwork.didentity)

    @staticmethod
    def sigmoid(z):
        """
        Sigmoid function
        """
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def dsigmoid(z):
        """
        Derivative of the sigmoid function
        """
        return NeuralNetwork.sigmoid(z) * (1.0 - NeuralNetwork.sigmoid(z))

    @staticmethod
    def identity(z):
        """
        Identity function
        """
        return z

    @staticmethod
    def didentity(z):
        """
        Derivative of the identity function
        """
        return np.ones_like(z)

    @staticmethod
    def mse_loss(y, t):
        """
        MSE loss
        """
        loss = 0.5 * np.mean((y - t) ** 2)
        return loss

    @staticmethod
    def cross_entropy_loss(y, t):
        """
        Cross-entropy loss
        """
        loss = -np.mean(t * np.log(y) + (1.0 - t) * np.log(1.0 - y))
        return loss

    def predict(self, x):
        """
        Predicts (forward pass) output y for input batch x

        Parameters
        ----------
        x : array
            Input batch of shape (d, N), where N is the number of patterns and
            d is the dimension

        Returns
        -------
        z : list of arrays
            List containing the preactivations of all layers for input batch x.
            z[i] is an array of shape (ni, N), with ni the number of neurons in
            layer i and N the number of patterns
        y : list of arrays
            List containing the activations of all layers for input batch x.
            y[i] is an array of shape (ni, N), with ni the number of neurons in
            layer i and N the number of patterns
        """
        z = []
        y = []
        # --- TO-DO block: loop in the network layers computing both the pre-
        # --- activation and the activation and appending them to lists z and y.
        pass
        # --- End of TO-DO block

        return z, y

    def compute_gradients(self, x, t):
        """
        Implements the backward pass, calculating the gradients of the loss
        function with respect to all the weights and biases for an input batch
        (x, t)

        Parameters
        ----------
        x : array
            Input batch of shape (d, N), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (1, N) with the target values for each x

        Returns
        -------
        db : list of arrays
            List containing the gradients of the loss function with respect to
            the biases of each layer, for input batch x.
        dW : list of arrays
            List containing the gradients of the loss function with respect to
            the weights of each layer, for input batch x.
        """
        n = x.shape[1]

        dW = []
        db = []

        # Call the predict method (forward pass). The preactivations z and the
        # activations y in each layer are needed for the backward pass:
        z, y = self.predict(x)

        # Derivative of the loss with respect to z in the last layer. This is
        # valid both for a regression problem with linear output and MSE loss,
        # and for a classification problem with sigmoid output and cross-entropy
        # loss:
        dy = (y[-1] - t) / n

        # --- TO-DO block: loop in the network layers computing the gradients with
        # --- respect to W and b. Note that the gradients must be computed starting
        # --- by the last layer, it may be useful to traverse the lists backwards.
        pass
        # --- End of TO-DO block

        return dW, db

    def gradient_step(self, x, t, eta):
        """
        Updates the model parameters with an input batch (x, t)

        Parameters
        ----------
        x : array
            Input batch of shape (d, N), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (1, N) with the target values for each x
        eta : float
            Learning rate
        """
        dW, db = self.compute_gradients(x, t)

        # --- TO-DO block: Loop in layers updating the model parameters b and w
        pass
        # --- End of TO-DO block

    def fit(self, x, t, eta, num_epochs, batch_size, loss_function):
        """
        Trains the model for a fixed number of epochs

        Parameters
        ----------
        x : array
            Input batch of shape (d, N), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (1, N) with the target values for each x
        eta : float
            Learning rate
        num_epochs : int
            Number of training epochs
        batch_size : int
            Size of mini-batches
        loss_function : function
            Loss function, either one of mse_loss or cross_entropy_loss

        Returns
        -------
        loss : array
             Array of shape (num_epochs,) with the loss after each training
             epoch
        """
        dim, n = x.shape
        num_batches = (n // batch_size) + ((n % batch_size) != 0)

        loss = np.zeros(num_epochs)
        for i in range(num_epochs):
            # Shuffle data and generate batches:
            ix = np.random.permutation(n)
            for j in range(num_batches):
                imin = j * batch_size
                imax = np.minimum((j + 1) * batch_size, n)

                ibatch = ix[imin:imax]
                batch_x = x[:, ibatch]
                batch_t = t[:, ibatch]
                self.gradient_step(batch_x, batch_t, eta)

            # At the end of each epoch, compute the model loss on all data:
            loss[i] = self.get_loss(x, t, loss_function)
        return loss

    def get_loss(self, x, t, loss_function):
        """
        Calculates the loss for an input batch (x, t)

        Parameters
        ----------
        x : array
            Input batch of shape (d, N), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (1, N) with the target values for each x
        loss_function : function
            Loss function, either one of mse_loss or cross_entropy_loss

        Returns
        -------
        loss : float
             Calculated loss
        """
        _, y = self.predict(x)
        return loss_function(y[-1], t)


class NeuralNetwork_TF(object):
    """
    Dense feedforward neural network model for exercise 7, using TensorFlow

    Parameters
    ----------
    d : list of tuples
        List of tuples that define the neural network architecture. There is one
        tuple of the form (n, a) for each layer, where n is the number of layer
        neurons and a is the activation. Any activation function in TensroFlow is
        valid. In particular:

        - tf.sigmoid: a layer of sigmoid units
        - tf.identity: a layer of linear units

        The activation function is ignored for the input layer.
        The minimum is two layers (input and output).
        The default value is layers=[(2, None), (1, tf.sigmoid)].
        The output layer must have only one neuron.

    Attributes
    ----------
    nlayers : int
        Number of layers, excluding the input layer
    W : list of tensors
        List containing the weight matrix for each processing layer
    b : list of tensors
        List containing the bias vector for each processing layer
    a : list of functions
        List containing the activation function for each processing layer
    """

    def __init__(self, layers=[(2, None), (1, tf.sigmoid)]):
        self.nlayers = len(layers) - 1
        self.W = []
        self.b = []
        self.a = []
        for l0, l1 in zip(layers[:-1], layers[1:]):
            self.W.append(
                tf.Variable(
                    tf.random.normal(shape=[l1[0], l0[0]], dtype=tf.dtypes.float64)
                )
            )
            self.b.append(
                tf.Variable(tf.random.normal(shape=[l1[0], 1], dtype=tf.dtypes.float64))
            )
            self.a.append(l1[1])

    @staticmethod
    def mse_loss(y, t):
        """
        MSE loss
        """
        loss = 0.5 * tf.reduce_mean(tf.pow(y - t, 2.0))
        return loss

    @staticmethod
    def cross_entropy_loss(y, t):
        """
        Cross-entropy loss
        """
        loss = -tf.reduce_mean(t * tf.math.log(y) + (1.0 - t) * tf.math.log(1.0 - y))
        return loss

    def predict(self, x):
        """
        Predicts (forward pass) output y for input batch x

        Parameters
        ----------
        x : tensor
            Input batch of shape (d, N), where N is the number of patterns and
            d is the dimension

        Returns
        -------
        y : tensor
            Tensor of shape (1, N) with the activation in the last layer
        """
        # --- TO-DO block: loop in the network layers computing the activations.
        # --- The activation of the last layer should be stored at variable y to
        # --- be returned.
        pass
        # --- End of TO-DO block

        return y

    def compute_gradients(self, x, t, loss_function):
        """
        Implements the backward pass, calculating the gradients of the loss
        function with respect to all the weights and biases for an input batch
        (x, t)

        Parameters
        ----------
        x : tensor
            Input batch of shape (d, N), where N is the number of patterns and
            d is the dimension
        t : tensor
            Tensor of shape (1, N) with the target values for each x
        loss_function : function
            Loss function, either one of mse_loss or cross_entropy_loss

        Returns
        -------
        db : list of tensors
            List containing the gradients of the loss function with respect to
            the biases of each layer, for input batch x.
        dW : list of tensors
            List containing the gradients of the loss function with respect to
            the weights of each layer, for input batch x.
        """
        # --- TO-DO block: compute the gradients db, dW using the gradient tape
        pass
        # --- End of TO-DO block

        return db, dW

    # ---------------------------------------------------------------------------
    # Gradient step:
    # ---------------------------------------------------------------------------
    def gradient_step(self, x, t, eta, loss_function):
        """
        Updates the model parameters with an input batch (x, t)

        Parameters
        ----------
        x : tensor
            Input batch of shape (d, N), where N is the number of patterns and
            d is the dimension
        t : tensor
            Tensor of shape (1, N) with the target values for each x
        eta : float
            Learning rate
        loss_function : function
            Loss function, either one of mse_loss or cross_entropy_loss
        """
        dB, dW = self.compute_gradients(x, t, loss_function)

        # --- TO-DO block: Loop in layers updating the model parameters b and w
        pass
        # --- End of TO-DO block

    def fit(self, x, t, eta, num_epochs, batch_size, loss_function):
        """
        Trains the model for a fixed number of epochs

        Parameters
        ----------
        x : array
            Input batch of shape (d, N), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (1, N) with the target values for each x
        eta : float
            Learning rate
        num_epochs : int
            Number of training epochs
        batch_size : int
            Size of mini-batches
        loss_function : function
            Loss function, either one of mse_loss or cross_entropy_loss

        Returns
        -------
        loss : array
             Array of shape (num_epochs,) with the loss after each training
             epoch
        """
        dim, n = x.shape
        num_batches = (n // batch_size) + ((n % batch_size) != 0)

        loss = np.zeros(num_epochs)
        for i in range(num_epochs):
            # Shuffle data and generate batches:
            ix = np.random.permutation(n)
            for j in range(num_batches):
                imin = j * batch_size
                imax = np.minimum((j + 1) * batch_size, n)

                ibatch = ix[imin:imax]
                batch_x = x[:, ibatch]
                batch_t = t[:, ibatch]
                self.gradient_step(batch_x, batch_t, eta, loss_function)

            # Calculo el loss de la epoca con todos los datos:
            loss[i] = self.get_loss(x, t, loss_function).numpy()
        return loss

    def get_loss(self, x, t, loss_function):
        """
        Calculates the loss for an input batch (x, t)

        Parameters
        ----------
        x : array
            Input batch of shape (d, N), where N is the number of patterns and
            d is the dimension
        t : array
            Array of shape (1, N) with the target values for each x
        loss_function : function
            Loss function, either one of mse_loss or cross_entropy_loss

        Returns
        -------
        loss : float
             Calculated loss
        """
        y = self.predict(x)
        return loss_function(y, t)
