# !IMPORTANT: 
# this script was tested only on tf-nightly==1.5.0.dev20171031 build

import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution()
context = tf.device('/cpu:0')
context.__enter__()

# download resnet_model
import sys, os, requests
resnet_model_url="https://raw.githubusercontent.com/tensorflow/models/master/official/resnet/resnet_model.py"
response = requests.get(resnet_model_url)
open("resnet_model.py", "wb").write(response.text)
sys.path.insert(0, ".")
import resnet_model


HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10
BATCH_SIZE=128
_WEIGHT_DECAY = 2e-4
_INITIAL_LEARNING_RATE = 0.1 * BATCH_SIZE / 128
_MOMENTUM = 0.9
RESNET_SIZE=32


from tensorflow.python.eager import graph_callable

images = tf.zeros((BATCH_SIZE, HEIGHT, WIDTH, DEPTH))
l = tf.cast(tf.random_uniform([BATCH_SIZE], maxval=NUM_CLASSES), tf.int32)
labels = tf.one_hot(l, NUM_CLASSES)

@graph_callable.graph_callable([])
def resnet_loss():
    """Resnet loss from random input"""
    network = resnet_model.cifar10_resnet_v2_generator(RESNET_SIZE, NUM_CLASSES)
    inputs = tf.reshape(images, [BATCH_SIZE, HEIGHT, WIDTH, DEPTH])
    logits = network(inputs,True)
    cross_entropy = tf.losses.softmax_cross_entropy(logits=logits,
                                                    onehot_labels=labels)
    return cross_entropy

loss_and_grads_fn = tfe.implicit_value_and_gradients(resnet_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
losses = []

for i in range(100):
  loss, grads_and_vars = loss_and_grads_fn()
  optimizer.apply_gradients(grads_and_vars)
  print(loss)
  losses.append(loss.numpy())

print("Final losses list: {}".format(losses))
