import numpy as np
from scipy.stats import logistic
import tensorflow as tf
import keras
import pandas as pd


class MAML(object):

    def __init__(self):
        # Number of data points in each data set
        self.num_samples = 50
        # Number of points in test set
        self.num_test_samples = 15
        # Number of training iterations
        self.epochs = 15
        # Hyperparameter for the inner loop (inner gradient update)
        self.alpha = 0.0001
        # Hyperparameter for the outer loop (outer gradient update)
        self.beta = 0.0001
        # Number of parameters to track
        self.n_params = 30

        self.data_set_list_x = []
        self.data_set_list_y = []
        self.n_data_sets_to_mock = 3
        self.n_rows_to_mock = 40
        self.n_cols_to_mock = 12
        self.num_tasks = self.n_data_sets_to_mock

        # MAML Algorithm 2: Line 1
        self.theta = np.random.normal(size=self.n_cols_to_mock).reshape(self.n_cols_to_mock, 1)

    def __sample_points(self, k):
        x = np.random.rand(k, self.n_cols_to_mock)
        y = np.random.choice([0, 1], size=k, p=[.5, .5]).reshape([-1, 1])
        return x, y

    def __compute_loss(self, f_phi, y):
        """Equation 3 from paper"""
        return ((np.matmul(-y.T, np.log(f_phi)) + np.matmul((1 - y.T), np.log(1 - f_phi))) / self.n_rows_to_mock).sum()

    def train(self):
        """Following Algorithm 2 from original MAML paper

        1. Initialize global parameter theta
        2. Enter loop and run fixed number of steps
        3. Enter loop through classification tasks
            4. Pull training data associated with current classification task
            5. Sample the points (we just use all of them)
            6. Evaluate the gradient of the loss function
        """
        # MAML Algorithm 2: Line 2
        for e in range(self.epochs):
            theta_prime = []
            # MAML Algorithm 2: Line 4
            for i in range(self.n_data_sets_to_mock):
                # MAML Algorithm 2: Lines 3 and 5
                x_i, y_i = self.data_set_list_x[i], self.data_set_list_y[i]
                # "Consider a model, denoted f, that maps observations x to outputs a"
                f_theta = logistic.cdf(np.matmul(x_i, self.theta))
                # Compute cross entropy loss function
                loss = self.__compute_loss(f_theta, y_i)
                # MAML Algorithm 2: Line 6
                gradient = np.matmul(x_i.T, (f_theta - y_i)) / self.n_rows_to_mock
                # MAML Algorithm 2: Line 7
                theta_prime.append(self.theta - self.alpha * gradient)

            # MAML Algorithm 2: Line 8
            meta_gradient = np.zeros(self.theta.shape)
            for i in range(self.n_data_sets_to_mock):
                # sample k data points and prepare our test set for meta training
                x_test, y_test = self.__sample_points(self.num_test_samples)
                # predict the value of y
                y_pred = logistic.cdf(np.matmul(x_test, theta_prime[i]))
                # compute meta gradients
                meta_gradient += np.matmul(x_test.T, (y_pred - y_test)) / self.n_rows_to_mock
            # MAML Algorithm 2: Line 10
            self.theta = self.theta - self.beta * meta_gradient / self.n_data_sets_to_mock
            if e % 1000 == 0:
                print("Epoch {}: Loss {}\n".format(e, loss))

    def mock_data(self):
        for n in range(self.n_data_sets_to_mock):
            self.data_set_list_x.append(np.random.rand(self.n_rows_to_mock, self.n_cols_to_mock))
            self.data_set_list_y.append(np.random.rand(self.n_rows_to_mock, 1))


class MyModelKeras(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the weights to `5.0` and the bias to `0.0`
        # In practice, these should be randomly initialized
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def call(self, x):
        return self.w * x + self.b


class KerasTrainer:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.weights = []
        self.biases = []
        self.epochs = range(10)
        self.TRUE_W = 3.0
        self.TRUE_B = 2.0
        self.NUM_EXAMPLES = 201
        self.model = None
        self._execute()

    def _execute(self):
        x = tf.linspace(-2, 2, self.NUM_EXAMPLES)
        self.x = tf.cast(x, tf.float32)
        f = lambda xl: xl * self.TRUE_W + self.TRUE_B
        noise = tf.random.normal(shape=[self.NUM_EXAMPLES])
        self.y = f(self.x) + noise

        self.model = MyModelKeras()
        self.training_loop()

    def training_loop(self):
        for epoch in self.epochs:
            # Update the model with the single giant batch
            self.train(learning_rate=0.1)

            # Track this before I update
            self.weights.append(self.model.w.numpy())
            self.biases.append(self.model.b.numpy())
            current_loss = self.loss(self.y, self.model(self.x))

            print(f"Epoch {epoch:2d}:")
            print("    ", self.report(self.model, current_loss))

    # This computes a single loss value for an entire batch
    @staticmethod
    def loss(target_y, predicted_y):
        return tf.reduce_mean(tf.square(target_y - predicted_y))

    # Given a callable model, inputs, outputs, and a learning rate...
    def train(self, learning_rate=0.1):
        with tf.GradientTape() as t:
            # Trainable variables are automatically tracked by GradientTape
            current_loss = self.loss(self.y, self.model(self.x))

        # Use GradientTape to calculate the gradients with respect to W and b
        dw, db = t.gradient(current_loss, [self.model.w, self.model.b])

        # Subtract the gradient scaled by the learning rate
        self.model.w.assign_sub(learning_rate * dw)
        self.model.b.assign_sub(learning_rate * db)

    # Define a training loop
    def report(self, model, current_loss):
        return f"W = {model.w.numpy():1.2f}, b = {model.b.numpy():1.2f}, loss={current_loss:2.5f}"


if __name__ == "__main__":
    # model = MAML()
    # model.mock_data()
    # model.train()

    '''df = pd.read_csv(
        '/Users/dfox/data/C800/vis-machine-learning/patient_response/data_dumps/C800_lab_data_OHE_12-22.csv')
    X = df.drop(columns=['identifier', 'clinical_benefit'])
    X_cols = X.columns
    y = df['clinical_benefit']
    '''
    X_tr = pd.read_csv('~/data/C800/vis-machine-learning/patient_response/notebooks/X_train.csv', index_col=0)
    X_te = pd.read_csv('~/data/C800/vis-machine-learning/patient_response/notebooks/X_test.csv', index_col=0)
    y_tr = pd.read_csv('~/data/C800/vis-machine-learning/patient_response/notebooks/y_train.csv', index_col=0)
    y_te = pd.read_csv('~/data/C800/vis-machine-learning/patient_response/notebooks/y_test.csv', index_col=0)

    #kt = KerasTrainer()
