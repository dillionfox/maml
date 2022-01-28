import tensorflow as tf
from tensorflow import keras
import pandas as pd
import autokeras as ak
from sklearn.metrics import accuracy_score


class ClinicalDataModel:
    def __init__(self, general_train_x, general_train_y, specific_train_x, specific_train_y, test_x, test_y):
        self.general_train_x = general_train_x
        self.general_train_y = general_train_y
        self.specific_train_x = specific_train_x
        self.specific_train_y = specific_train_y
        self.test_x = test_x
        self.test_y = test_y
        self.pretrained_optimizer = None
        self.meta_trained_optimizer = None
        self.train_accuracy = None
        self.model = None
        self._build_model()

    def __str__(self):
        return """
        This class stores the data required for Model Agnostic Meta-Learning
        
        There are 3 types of data sets. 
        
        general_train_x/y: The large pool of non-specific training data, i.e., a whole bunch of clinical trials.
        specific_train_x/y: The one clinical trial dataset that you're interested in.
        test_x/y: Some held out data from the clinical trial you're interested in. 
        
        Autokeras is used to build a generic neural network. Autokeras figures out how many layers to use
        and how to parameterize each layer. It's not the best, but it's fine for now.
        
        """

    def _build_model(self):
        clf = ak.StructuredDataClassifier(overwrite=True, max_trials=3)
        clf.fit(x=self.general_train_x, y=self.general_train_y, epochs=10)
        predicted_y = clf.predict(self.test_x)
        self.train_accuracy = accuracy_score(self.test_y, predicted_y)
        clf.evaluate(x=self.test_x, y=self.test_y)
        self.model = clf.export_model()
        self.weights = self.model.trainable_weights


class MAML:
    def __init__(self, clinical_data):
        self.clinical_data = clinical_data
        self.alpha = 0.001
        self.beta = 0.001
        self.meta_train_iterations = 1000
        self.meta_batch_size = tf.cast(25.0, dtype=tf.float32)
        self.num_updates = 5
        self.num_epochs = 100
        self.optimizer = keras.optimizers.SGD
        self.train_accuracy = None
        self.test_accuracy = None
        self.train_loss = None
        self.test_loss = None
        self.pretrain_op = None
        self.meta_train_optimized = None
        self._meta_learn_out_type = None
        self._preprocess()

    def execute(self):

        # Execute outer loop
        for epoch in range(self.num_epochs):
            # Prepare feed dict: feed_dict = {model.inputa: inputa, model.inputb: inputb, ...
            # Prepare input_tensors: self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)
            # Run one step: result = sess.run(input_tensors, feed_dict)

            # Chunk up the data set into meta_batch_size sets and run them all.
            results = tf.map_fn(self.meta_learn_step,
                                elems=(self.clinical_data.train_x, self.clinical_data.train_y,
                                       self.clinical_data.test_x, self.clinical_data.test_y),
                                dtype=self._meta_learn_out_type,
                                parallel_iterations=self.meta_batch_size)

            # Unpack results
            self.train_loss, self.test_loss = results[0], results[1]
            self.train_accuracy, self.test_accuracy = results[2], results[3]

        self.report_metrics()

    def meta_learn_step(self, data_sets_list):
        train_forward_n = None
        test_loss = []
        train_x, train_y, test_x, test_y = data_sets_list
        weights = self.clinical_data.weights
        train_forward_1 = self.forward_propagation(weights, reuse=False)  # only reuse on the first iter
        train_loss = self.loss_func(train_forward_1, train_y)

        # Compute accuracy on training data
        train_accuracy = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(train_forward_1), 1),
                                                     tf.argmax(train_y, 1))

        # Initialize theta
        theta, _, __ = self.compute_grads(test_x, test_y, weights, train_loss, self.alpha)

        for j in range(self.num_updates - 1):
            loss = self.loss_func(self.forward_propagation(train_x, theta, reuse=True), train_y)
            theta, train_forward_n, test_loss = self.compute_grads(test_x, test_y, theta, loss, self.alpha)

        test_accuracy = [tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(train_forward_n[j]), 1),
                                                     tf.argmax(test_y, 1)) for j in range(self.num_updates)]

        return train_loss, test_loss, train_accuracy, test_accuracy

    @staticmethod
    def loss_func(targets, predictions):
        return keras.losses.BinaryCrossentropy(from_logits=True)(targets, predictions)

    def report_metrics(self):
        total_loss_train = tf.reduce_sum(self.train_loss) / self.meta_batch_size
        total_loss_test = [tf.reduce_sum(self.test_loss[j]) / self.meta_batch_size for j
                           in range(self.num_updates)]

        total_accuracy_train = tf.reduce_sum(self.train_accuracy) / self.meta_batch_size
        total_accuracy_test = [tf.reduce_sum(self.test_accuracy[j]) / self.meta_batch_size for j in
                               range(self.num_updates)]

        tf.summary.scalar('Pre-update loss', total_loss_train)
        tf.summary.scalar('Pre-update accuracy', total_accuracy_train)

        for j in range(self.num_updates):
            tf.summary.scalar('Post-update loss, step ' + str(j + 1), total_loss_test[j])
            tf.summary.scalar('Post-update accuracy, step ' + str(j + 1), total_accuracy_test[j])

        # Extra metrics
        self.pretrain_op = self.optimizer(self.beta).minimize(total_loss_train)
        optimizer = self.optimizer(self.beta)
        gvs = optimizer.compute_gradients(total_loss_test[self.num_updates - 1])
        self.meta_train_optimized = optimizer.apply_gradients(gvs)

    def compute_grads(self, test_x, test_y, weights, loss, alpha):
        train_forward_n, train_forward_loss_n = [], []
        grads = tf.gradients(loss, list(weights.values()))
        gradients = dict(zip(weights.keys(), grads))
        theta = dict(zip(weights.keys(), [weights[key] - alpha * gradients[key] for key in weights.keys()]))
        output = self.forward_propagation(test_x, theta, reuse=True)
        train_forward_n.append(output)
        train_forward_loss_n.append(self.loss_func(output, test_y))
        return theta, train_forward_n, train_forward_loss_n

    @property
    def dim_hidden(self):
        return []

    def forward_propagation(self, inp, weights, reuse=False):
        """ Compute the network output """
        hidden = self.normalize(tf.matmul(inp, weights['w1']) + weights['b1'])
        for i in range(1, len(self.dim_hidden)):
            hidden = self.normalize(tf.matmul(hidden, weights['w' + str(i + 1)]) + weights['b' + str(i + 1)])
        return tf.matmul(hidden, weights['w' + str(len(self.dim_hidden) + 1)]) + weights[
            'b' + str(len(self.dim_hidden) + 1)]

    @staticmethod
    def normalize(inp):
        return keras.layers.BatchNormalization(inp)

    def _preprocess(self):
        self._meta_learn_out_type = [tf.float32, [tf.float32] * self.num_updates,
                                     [tf.float32, [tf.float32] * self.num_updates]]


def generate_fake_datasets(x_df, y):
    x_new = []
    y_new = []
    mu = 0
    sigma = 0.01
    for i in range(3):
        train_noise = pd.DataFrame(np.random.normal(mu, sigma, x_df.shape), columns=x_df.columns)
        x_new.append(x_df.reset_index(drop=True) + train_noise)
        y_new.append(y.sample(frac=1).reset_index(drop=True))
    x_set = pd.concat(x_new)
    y_set = pd.concat(y_new)
    return x_set, y_set


if __name__ == "__main__":
    # Load prepared datasets. Most data is
    X_train = pd.read_csv('~/data/C800/vis-machine-learning/patient_response/notebooks/X_train.csv', index_col=0)
    y_train = pd.read_csv('~/data/C800/vis-machine-learning/patient_response/notebooks/y_train.csv', index_col=0)

    X_test = pd.read_csv('~/data/C800/vis-machine-learning/patient_response/notebooks/X_test.csv', index_col=0)
    y_test = pd.read_csv('~/data/C800/vis-machine-learning/patient_response/notebooks/y_test.csv', index_col=0)

    # Generate synthetic training datasets
    synthetic_train_x, synthetic_train_y = generate_fake_datasets(X_train, y_train)

    # Construct object to store data and auto-build basic keras model
    clinical_data_model = ClinicalDataModel(synthetic_train_x, synthetic_train_y,
                                            X_train, y_train,
                                            X_test, y_test)
