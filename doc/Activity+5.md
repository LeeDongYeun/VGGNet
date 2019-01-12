
# MNIST Hand-written digit classification

### **2018/11/27 CoE 202 Activity 5**<br/>
Authorized by SIIT, KAIST
Yekang Lee, Jaemyung Yu, and Junmo Kim<br/>

***Tip> shotcuts for Jupyter Notebook***
* Shift + Enter : run cell and select below

***Library***
* Numpy: Fundamenta package for scientific computing with Python
* Tensorflow: An open source machine learning library for research and production
* Matplotlib: Python 2D plottin glibrary


```python
from __future__ import print_function
from collections import namedtuple
from functools import partial

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.training import moving_averages

tf.logging.set_verbosity(tf.logging.ERROR)
```

## 1. Prepare the data
### <a href=http://yann.lecun.com/exdb/mnist/>MNIST dataset</a>
The MNIST has a training set of 55,000 examples, a validation set of 5,000 examples and a test set of 10,000 examples.


```python
mnist = input_data.read_data_sets('./data/', one_hot=True)
```

    Extracting ./data/train-images-idx3-ubyte.gz
    Extracting ./data/train-labels-idx1-ubyte.gz
    Extracting ./data/t10k-images-idx3-ubyte.gz
    Extracting ./data/t10k-labels-idx1-ubyte.gz

<div style="page-break-after: always;"></div>

Load the training dataset


```python
train_images = mnist.train.images
train_labels = mnist.train.labels
train_images = train_images.reshape([-1, 28, 28, 1])
```

Load the validation sets


```python
val_images = mnist.validation.images
val_labels = mnist.validation.labels
val_images = val_images.reshape([-1, 28, 28, 1])
```

Plot the 1st hand-written digit and its one-hot label


```python
plt.imshow(train_images[0,:,:,0], cmap='Greys')
print("\nOne-hot labels for this image:")
print(train_labels[0])
```


    One-hot labels for this image:
    [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]



![png](output_9_1.png)

<div style="page-break-after: always;"></div>

## 2. Build a graph

Very Deep Convolutional Networks for Large-scale image recognition
https://arxiv.org/abs/1409.1556

<img src="../img/fig2.png">

### Set hyperparameters
- ***log_dir*** : Directory name to save models
- ***n_epochs*** : Maximun training epoch
- ***n_outputs*** : The number of classes for labels
- ***init_lr*** : Learning rate for gradient descent
- ***l2_lambda*** : regularization parameter
- ***batch_size*** : The number of images to update paramerters once


```python
log_dir = 'logs/'
n_epochs = 20
n_outputs = 10
init_lr = 0.01
batch_size = 100
l2_lambda = 0.0001
```

### Placeholder for learning rate, input images and labels


```python
lrn_rate = tf.placeholder(tf.float32, shape=(), name='lrn_rate')
images = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='images')
labels = tf.placeholder(tf.int32, shape=(None), name='labels')
```


```python
def vggnet(images, labels=None):
    vggnet_conv = partial(tf.layers.conv2d, kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_lambda), padding="SAME")
    
    ''' 1st conv. '''
    x = vggnet_conv(images, filters=16, kernel_size=7, strides=[1,1])  # 7x7 filter, # of filters: 16, stride: 1
    x = tf.layers.batch_normalization(x, name='bn1')                   # batch normalization
    x = tf.nn.relu(x)                                                  # ReLU activation
    
    ''' 2nd conv.'''
    x = vggnet_conv(x, filters=16, kernel_size=3, strides=[1,1])       # 3x3 filter, # of filters: 16, stride 1
    x = tf.layers.batch_normalization(x, name='bn2')                   # batch normalization
    x = tf.nn.relu(x)                                                  # ReLU activation
    x = tf.layers.average_pooling2d(x, pool_size=[2,2], strides=[2,2]) # 2x2 average pooling, stride: 2
    
    ''' 3rd conv. '''
    x = vggnet_conv(x, filters=32, kernel_size=3, strides=[1, 1])      # 3x3 filter, # of filters: 32, stride: 1
    x = tf.layers.batch_normalization(x, name='bn3')                   # batch normalization
    x = tf.nn.relu(x)                                                  # ReLU activation
    
    ''' 4th conv. '''
    x = vggnet_conv(x, filters=32, kernel_size=3, strides=[1,1])       # 3x3 filter, # of filters: 32, stride 1
    x = tf.layers.batch_normalization(x, name='bn4')                   # batch normalization
    x = tf.nn.relu(x)                                                  # ReLU activation
    x = tf.layers.average_pooling2d(x, pool_size=[2,2], strides=[2,2]) # 2x2 average pooling stride 2
    
    ''' 5th conv. '''
    x = vggnet_conv(x, filters=64, kernel_size=3, strides=[1,1])       # 3x3 filter, # of filters: 64, stride: 1
    x = tf.layers.batch_normalization(x, name='bn5')                   # batch normalization
    x = tf.nn.relu(x)                                                  # ReLU activation
    
    ''' 6th conv. '''
    x = vggnet_conv(x, filters=64, kernel_size=3, strides=[1,1])       # 3x3 filter, # of filters: 64, stride: 1
    x = tf.layers.batch_normalization(x, name='bn6')                   # batch normalization
    x = tf.nn.relu(x)                                                  # ReLU activation
    
    img_feat = tf.reduce_mean(x, [1, 2])                               # Global average pooling
    return img_feat
```

### Build a model


```python
global_step = tf.Variable(0, trainable=False)

with tf.variable_scope('embed') as scope:
#     feats = simple_network(images)
    feats = vggnet(images)

## Reshape
feats = tf.reshape(feats, [batch_size, 64])
    
## Logits
logits = tf.layers.dense(feats, n_outputs, kernel_initializer=tf.uniform_unit_scaling_initializer(factor=2.0), 
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_lambda))

## Evaluation
correct = tf.nn.in_top_k(logits, labels, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

## SOFTMAX
preds = tf.nn.softmax(logits)

## Cost function
cent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
cost_cls = tf.reduce_mean(cent, name='cent')
```

### L2 regularization


```python
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
cost = tf.add_n([cost_cls] + reg_losses)
```

### Momentum optimizer


```python
lr  = tf.train.exponential_decay(init_lr, global_step, 1000, 0.8, staircase = True) # learning rate decay
optimizer = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)                  # Momentum optimizer
train_op = optimizer.minimize(cost)
```

<div style="page-break-after: always;"></div>

# 3. Train a model

### Create a session and initialize parameters
Tensorflow operations must be executed in the session.


```python
## MAKE SESSION
sess = tf.Session()

## INITIALIZE SESSION
sess.run(tf.global_variables_initializer())
```

### Updates parameters with back-propagation


```python
for epoch in range(n_epochs+1):
    for iteration in range(mnist.train.num_examples // batch_size):
        start_time = time.time()
        
        X_batch, y_batch = mnist.train.next_batch(batch_size)
        X_batch = X_batch.reshape([-1, 28, 28, 1])
        (_, loss, loss_cls, prediction) = sess.run([train_op, cost, cost_cls, preds], 
                                                    feed_dict={images: X_batch, labels: y_batch})
        duration = time.time() - start_time
        sec_per_batch = float(duration)
    
    ## Training accuracy every one epoch
    acc_train = accuracy.eval(session=sess, feed_dict={images: X_batch, labels: np.argmax(y_batch, axis=1)})
    if epoch % 1 == 0:
        print('  [*] TRAINING Iteration %d, Loss: %.4f, Acc: %.4f (duration: %.3fs)'
                             % (epoch, loss_cls, acc_train, sec_per_batch))

    ## Validation accuracy every 5 epochs
    if epoch % 5 == 0:
        acc_val = accuracy.eval(session=sess, feed_dict={images: val_images, labels: np.argmax(val_labels, axis=1)})
        print('  [*] VALIDATION ACC: %.3f' % acc_val)

print('Optimization done.')
```

      [*] TRAINING Iteration 0, Loss: 0.1849, Acc: 0.9700 (duration: 0.006s)
      [*] VALIDATION ACC: 0.920
      [*] TRAINING Iteration 1, Loss: 0.1961, Acc: 0.9500 (duration: 0.007s)
      [*] TRAINING Iteration 2, Loss: 0.0423, Acc: 1.0000 (duration: 0.007s)
      [*] TRAINING Iteration 3, Loss: 0.0995, Acc: 1.0000 (duration: 0.007s)
      [*] TRAINING Iteration 4, Loss: 0.1445, Acc: 0.9600 (duration: 0.006s)
      [*] TRAINING Iteration 5, Loss: 0.0816, Acc: 0.9900 (duration: 0.007s)
      [*] VALIDATION ACC: 0.984
      [*] TRAINING Iteration 6, Loss: 0.0155, Acc: 1.0000 (duration: 0.006s)
      [*] TRAINING Iteration 7, Loss: 0.0051, Acc: 1.0000 (duration: 0.006s)
      [*] TRAINING Iteration 8, Loss: 0.0146, Acc: 1.0000 (duration: 0.007s)
      [*] TRAINING Iteration 9, Loss: 0.0406, Acc: 0.9900 (duration: 0.008s)
      [*] TRAINING Iteration 10, Loss: 0.0262, Acc: 1.0000 (duration: 0.007s)
      [*] VALIDATION ACC: 0.985
      [*] TRAINING Iteration 11, Loss: 0.0054, Acc: 1.0000 (duration: 0.006s)
      [*] TRAINING Iteration 12, Loss: 0.0490, Acc: 0.9900 (duration: 0.007s)
      [*] TRAINING Iteration 13, Loss: 0.0846, Acc: 0.9800 (duration: 0.007s)
      [*] TRAINING Iteration 14, Loss: 0.0222, Acc: 1.0000 (duration: 0.008s)
      [*] TRAINING Iteration 15, Loss: 0.0480, Acc: 0.9800 (duration: 0.007s)
      [*] VALIDATION ACC: 0.989
      [*] TRAINING Iteration 16, Loss: 0.0027, Acc: 1.0000 (duration: 0.006s)
      [*] TRAINING Iteration 17, Loss: 0.0088, Acc: 1.0000 (duration: 0.006s)
      [*] TRAINING Iteration 18, Loss: 0.0426, Acc: 0.9900 (duration: 0.006s)
      [*] TRAINING Iteration 19, Loss: 0.0454, Acc: 0.9900 (duration: 0.007s)
      [*] TRAINING Iteration 20, Loss: 0.0203, Acc: 1.0000 (duration: 0.006s)
      [*] VALIDATION ACC: 0.990
    Optimization done.

<div style="page-break-after: always;"></div>

# 4. Test a model

### Load the test images and labels


```python
## READ MNIST INPUTS
test_images = mnist.test.images
test_labels = mnist.test.labels
test_images = test_images.reshape([-1, 28, 28, 1])

## Plot the 1st test image and label
plt.imshow(test_images[0,:,:,0], cmap='Greys')
print("\nOne-hot labels for this image:")
print(test_labels[0])
```


    One-hot labels for this image:
    [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]



![png](output_29_1.png)


### Check the prediction for the first image


```python
prediction = sess.run(preds, feed_dict={images: test_images[0,:,:,0].reshape(1,28,28,1), labels: test_labels[0]})

print("The prediction of the network is: %d" % np.argmax(prediction))
```

    The prediction of the network is: 7

<div style="page-break-after: always;"></div>

### Average the accuray for test set


```python
test_acc = accuracy.eval(session=sess, feed_dict={images: test_images, labels: np.argmax(test_labels, axis=1)})
print('Acc: %.3f' % test_acc)
```

    Acc: 0.990

<div style="page-break-after: always;"></div>

## Assignment

### 1. A simple network for MNIST
Design the following CNN and apply the CNN to the MNIST dataset.
<img src="../img/fig3.png">

***Hint)*** 
- Elementwise-sum: tf.add() <br/>
- Concatenate: tf.concat()


### Submission (Due: Dec. 4 Tue.)
Compare the results with that of VGGNET and submit your report by Tuesday, Dec. 4 to <a href="mailto:kaiser5072@kaist.ac.kr">kaiser5072@kaist.ac.kr</a>

