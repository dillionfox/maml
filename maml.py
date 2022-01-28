import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import autokeras as ak
from sklearn.metrics import accuracy_score
from copy import deepcopy


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
        self.theta = self.model.trainable_weights

    @property
    def weights(self):
        return self.model.trainable_weights

    def make_new_model(self):
        return deepcopy(self)


class MAML:
    def __init__(self, clinical_data):
        self.clinical_data = clinical_data
        self.alpha = 0.001
        self.beta = 0.001
        self.meta_train_iterations = 1000
        self.meta_batch_size = tf.cast(25.0, dtype=tf.float32)
        self.num_updates = 5
        self.num_epochs = 100
        self._optimizer = keras.optimizers.SGD
        self._loss_func = keras.losses.BinaryCrossentropy(from_logits=True)
        self.train_accuracy = None
        self.test_accuracy = None
        self.train_loss = None
        self.test_loss = None
        self.pretrain_op = None
        self.meta_train_optimized = None
        self._meta_learn_out_type = None
        self._preprocess()

    def execute(self):

        optimizer = self._optimizer()
        theta = self.clinical_data.theta

        # Execute outer loop
        for epoch in range(self.num_epochs):

            with tf.GradientTape() as tape:
                # Chunk up the data set into meta_batch_size sets and run them all.
                results = tf.map_fn(self._meta_learn_step,
                                    elems=(self.clinical_data.train_x, self.clinical_data.train_y,
                                           self.clinical_data.test_x, self.clinical_data.test_y,
                                           theta),
                                    dtype=self._meta_learn_out_type,
                                    parallel_iterations=self.meta_batch_size)

                # Unpack results
                train_loss, test_loss = results[0], results[1]
                train_accuracy, test_accuracy = results[2], results[3]
                sum_losses_theta_prime = sum(results[4])

            # Get gradients of loss wrt the weights.
            gradients = np.array(tape.gradient(train_loss, sum_losses_theta_prime))

            if epoch == 0:
                theta = self.beta * gradients
            else:
                theta -= self.beta * gradients

            # Update the weights of the model - this needs to operate on a model object
            optimizer.apply_gradients(zip(gradients, sum_losses_theta_prime))

        self.report_metrics()

    def simpler_learn_step(self, data_sets_list):
        # Pull out the chunks of data passed through the map function.
        train_x, train_y, test_x, test_y, theta = data_sets_list

        # Save unmodified copy of training weights. We need this for a gradient later on.
        theta = self.clinical_data.weights

        # Make a deep copy of the model object. We're doing this in a parallel map, so we
        # need to be careful not to have multiple workers operating on the same object.
        model = self.clinical_data.make_new_model()

        # Create new reference to loss function. We don't want any confusion. There are multiple
        # loss function instances in this algorithm.
        loss_fn = self._loss_func

        # Instantiate the optimizer with the 'alpha' parameter from MAML Algorith 2, Line 6.
        optimizer_pre_update = self._optimizer(learning_rate=self.alpha)

        # Prepare to compute gradient of loss function w.r.t. theta. MAML Algorithm 2, line 5.
        # Move model forward and "tape"/record what happens, so detailed results can be referenced.
        with tf.GradientTape() as tape:
            # Forward propagate the model. This does not update weights.
            forward_prop_pre_update = model(train_x)
            # Compute loss function from one step
            pre_update_loss = loss_fn(train_y, forward_prop_pre_update)

        # MAML Algorith 2, Line 5.
        # Stop recording / exit context manager. Reference "tape" to compute the gradient.
        gradient_f_theta = tape.gradient(pre_update_loss, model.trainable_weights)

        # This line is tricky. Update 'model' by applying gradient. Now f(theta) --> f(theta').
        # Note. The parameter 'Alpha' is accounted for by the optimizer as the learning_rate.
        optimizer_pre_update.apply_gradients(zip(gradient_f_theta, model.trainable_weights))

        # MAML Algorithm 2, line 8.
        # Prepare to compute gradient of loss function w.r.t. theta'.
        with tf.GradientTape() as tape:
            # Forward propagate the model
            forward_prop_meta = model(train_x)
            # Compute loss function from one step
            meta_update_loss = loss_fn(train_y, forward_prop_meta)

        # MAML Algorith 2, Line 10.
        gradient_f_theta_prime = tape.gradient(meta_update_loss, theta)

        return gradient_f_theta_prime

    def _meta_learn_step(self, data_sets_list):
        model = keras.models.clone_model(self.clinical_data.model)
        train_x, train_y, test_x, test_y = data_sets_list
        loss_fn = self._loss_func
        optimizer = self._optimizer
        train_forward_prop = model(train_x)  # Forward pass
        train_loss = loss_fn(train_y, train_forward_prop)
        train_accuracy = accuracy_score(train_y, np.argmax(train_forward_prop.numpy(), axis=1))

        with tf.GradientTape() as tape:
            gradients = tape.gradient(train_loss, self.clinical_data.weights)
            optimizer.apply_gradients(zip(gradients, self.clinical_data.weights))

        # !! Need to figure out which pieces of gradients I want to keep.
        # Initialize theta
        theta = self.alpha * np.array(gradients)

        # Forward propagate model
        model.compile(optimizer=optimizer,
                      loss=loss_fn,
                      metrics=[keras.metrics.BinaryAccuracy()])
        training_results = model.fit(x=train_x, y=train_y, epochs=self.num_updates)

        # I know this doesn't work. It's close-ish. But this part is going to be difficult.
        out = model.output
        out._id = 0
        with tf.GradientTape() as tape:
            gradients = tape.gradient(out, model.trainable_weights)
        test_accuracy = training_results['binary_accuracy']
        test_loss = training_results['loss']

        theta -= self.alpha * gradients

        return train_loss, test_loss, train_accuracy, test_accuracy, theta

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
        self.pretrain_op = self._optimizer(self.beta).minimize(total_loss_train)
        optimizer = self._optimizer(self.beta)
        gvs = optimizer.compute_gradients(total_loss_test[self.num_updates - 1])
        self.meta_train_optimized = optimizer.apply_gradients(gvs)

    def _preprocess(self):
        self._meta_learn_out_type = [tf.float32, [tf.float32] * self.num_updates,
                                     [tf.float32, [tf.float32] * self.num_updates],
                                     [tf.float32] * len(self.clinical_data.model.layers)]


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
