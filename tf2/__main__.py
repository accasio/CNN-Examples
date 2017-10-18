import tensorflow as tf
import numpy as np
import utils as utils
import prettytensor as pt
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets('data/MNIST/', one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)

img_size = 28  # 28px
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1  # gray-scale
num_classes = 10 # 10 digits

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
x_pretty = pt.wrap(x_image)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)

weights_conv1 = utils.get_weights_variable(layer_name='layer_conv1')
weights_conv2 = utils.get_weights_variable(layer_name='layer_conv2')

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 64

# Counter for total number of iterations performed so far.
total_iterations = 0

# Split the test-set into smaller batches of this size.
test_batch_size = 256

utils.print_test_accuracy()
utils.optimize(num_iterations=1)
utils.print_test_accuracy()
utils.optimize(num_iterations=99) # We already performed 1 iteration above.
utils.print_test_accuracy(show_example_errors=True)
utils.optimize(num_iterations=900) # We performed 100 iterations above.
utils.print_test_accuracy(show_example_errors=True)
