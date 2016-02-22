# -*- coding: utf-8 -*-
from pamap import PAMAP
import tensorflow as tf
import numpy as np

# Converts a scalar to a one-hot encoding vector
def one_hot_encode ( range, value ):
	vec = np.zeros( range )
	vec[ value ] = 1
	return vec

# Loads the already-processed PAMAP2 dataset.
# In this processed dataset, features are a statistical summary
# resulting from the FFT of a 512-instances sliding window.
pamap = PAMAP( PAMAP.PROCESSED )

# Loads the LOSO-CV (leave-one-subject-out cross validation for
# one fold.
# @TODO So far only fold 1 (subject 1) is considered, in the future
# the model should load all folds.
data  = pamap.cross_validation( 1 )
train = data[ 'train' ]
test  = data[ 'test'  ]

# Starts the TensorFlow interactive session.
sess = tf.InteractiveSession()

# Placeholders will be filled with data from the PAMAP dataset.
# Shapes specify the dimensions of data inputs and outputs.
x  = tf.placeholder( tf.float32, shape=[ None, 280 ] )		# Number of features: 
y_ = tf.placeholder( tf.float32, shape=[ None, 12  ] )  	# Number of classes: 12

# Variables define the model parameters, i.e., the weights
# and biases in the computation graph.
W  = tf.Variable( tf.zeros( [ 280, 12 ] ) )					# Model weights
b  = tf.Variable( tf.zeros( [ 12      ] ) )					# Model biases

# Initializes all the model variables.
sess.run( tf.initialize_all_variables() )

# Computes the predicted class using the regression model.
# This can be done by matrix-multiplying the weights and the inputs and
# adding up the biases vector.
y = tf.nn.softmax( tf.matmul( x, W ) + b )

# Defines the cost function to be minimized.
cost_function = -tf.reduce_sum( y_ * tf.log(y) )			# Cross entropy

# Defines the optimization algorithm.
# In this case, the chosen optimization algorithm is the 
# gradient descent.
alpha = 0.01
train_step = tf.train.GradientDescentOptimizer( alpha ).minimize( cost_function )		# Gradient descent

# Runs the training stage.
# In each iteration, only a random batch of the training
# set will be considered for efficiency purposes.
iters = 1000
for i in range(iters):
	batch = pamap.random_sample( train, 0.05 )
	train_step.run( feed_dict = {
		x  : [                     s[ 'features' ]   for s in batch ],
		y_ : [ one_hot_encode( 12, s[ 'class'    ] ) for s in batch ]
	})

# Defines the function that establishes whether a prediction is correct.
# Also, defines the function used for measuring the classifier accuracy.
correct_prediction = tf.equal( tf.argmax( y, 1 ), tf.argmax( y_, 1 ) )
accuracy = tf.reduce_mean( tf.cast( correct_prediction, tf.float32 ) )

# Computes the accuracy of the classifier using the test set.
model_accuracy = accuracy.eval( feed_dict = {
	x  : [                     s[ 'features' ]   for s in test ],
	y_ : [ one_hot_encode( 12, s[ 'class'    ] ) for s in test ]
})
print model_accuracy



