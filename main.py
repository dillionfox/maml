import numpy as np
import random
import tensorflow as tf
from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags

summary_interval = 100
save_interval = 1000
print_interval = 100
test_print_interval = 100
metatrain_iterations = 1000

FLAGS = flags.FLAGS

# Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 2, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

# Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000,
                     'number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5,
                     'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.')  # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')

# Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

# Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1,
                     'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1,
                   'value of inner gradient step step during training. (use if you want to test with a different value)')  # 0.1 for omniglot
num_updates = 1
update_batch_size = 5


def train(model, saver, sess, exp_string):
    train_writer = tf.summary.FileWriter('./' + exp_string, sess.graph)

    # Meta-training iterations
    for itr in range(metatrain_iterations):

        # Create empty dict
        feed_dict = {}
        input_tensors = [model.metatrain_op]

        if itr % summary_interval == 0:
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[num_updates - 1]])
            input_tensors.extend([model.total_accuracy1, model.total_accuracies2[num_updates - 1]])
            result = sess.run(input_tensors, feed_dict)

            train_writer.add_summary(result[1], itr)
            print('Pretrain ' + str(itr) + ': ' + str(result[-2]) + ', ' + str(result[-1]))
            saver.save(sess, './' + exp_string + '/model' + str(itr))

            input_tensors = [model.total_accuracy1, model.total_accuracies2[num_updates - 1]]
            result = sess.run(input_tensors, feed_dict)
            print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

        saver.save(sess, './' + exp_string + '/model' + str(itr))


def make_feed_dict(model, batch_x, batch_y, num_classes):
    input_a = batch_x[:, :num_classes * update_batch_size, :]
    input_b = batch_x[:, num_classes * update_batch_size:, :]
    label_a = batch_y[:, :num_classes * update_batch_size, :]
    label_b = batch_y[:, num_classes * update_batch_size:, :]
    return {model.inputa: input_a, model.inputb: input_b, model.labela: label_a, model.labelb: label_b,
            model.meta_lr: 0.0}


def make_tensor_dict(ten, label_tensor, num_classes):
    return {'inputa': tf.slice(ten, [0, 0, 0], [-1, num_classes * update_batch_size, -1]),
            'inputb': tf.slice(ten, [0, num_classes * update_batch_size, 0], [-1, -1, -1]),
            'labela': tf.slice(label_tensor, [0, 0, 0],
                               [-1, num_classes * update_batch_size, -1]),
            'labelb': tf.slice(label_tensor, [0, num_classes * update_batch_size, 0],
                               [-1, -1, -1])}


def main():
    test_num_updates = 1
    data_generator = DataGenerator(10, 20)

    dim_output = data_generator.dim_output
    dim_input = data_generator.dim_input
    num_classes = data_generator.num_classes

    # Make tensors for training and meta evaluation
    random.seed(5)
    ten, label_tensor = data_generator.make_data_tensor()
    input_tensors = make_tensor_dict(ten, label_tensor, num_classes)
    random.seed(6)
    ten, label_tensor = data_generator.make_data_tensor(train=False)
    metaval_input_tensors = make_tensor_dict(ten, label_tensor, num_classes)

    # Construct model, train model and perform meta evaluations
    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')

    # Collect summary and save state
    model.summ_op = tf.summary.merge_all()
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)
    sess = tf.InteractiveSession()

    exp_string = 'sample_str'
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()
    train(model, saver, sess, exp_string, data_generator)


if __name__ == "__main__":
    main()
