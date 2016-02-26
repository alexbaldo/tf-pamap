# -*- coding: utf-8 -*-
from pamap import PAMAP
import tensorflow as tf
import numpy as np

# Converts a scalar to a one-hot encoding vector.
def one_hot_encode ( range, value ):
	vec = np.zeros( range )
	vec[ value ] = 1
	return vec

# Normalizes a vector to have zero mean and unit variance.
def normalize ( vector, means, variances ):
	return (vector - means) / variances

# Initializes non-zero weights for a given shape.
def weight_variable( shape ):
  initial = tf.truncated_normal( shape, stddev=0.1 )
  return tf.Variable( initial )

# Initializes non-zero biases for a given shape.
def bias_variable( shape ):
  initial = tf.constant( 0.1, shape=shape )
  return tf.Variable( initial )

# Initializes a 2D convolution operator.
def conv2d( x, W ):
  return tf.nn.conv2d( x, W, strides=[1, 1, 1, 1], padding='SAME' )

# Initializes a 2x1 pooling operator.
def max_pool_2x1( x ):
  return tf.nn.max_pool( x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME' )

# Loads the already-processed PAMAP2 dataset.
# In this processed dataset, features are a statistical summary
# resulting from the FFT of a 512-instances sliding window.
pamap = PAMAP( PAMAP.PROCESSED )

# Starts the TensorFlow interactive session.
sess = tf.InteractiveSession()

# Placeholders will be filled with data from the PAMAP dataset.
# Shapes specify the dimensions of data inputs and outputs.
x  = tf.placeholder( tf.float32, shape=[ None, 280 ] )		# Number of features: 
y_ = tf.placeholder( tf.float32, shape=[ None, 12  ] )  	# Number of classes: 12

# Reshapes the input data to a more accurate representation
# of the original data structure.
# Width: 40 sensors
# Height: 7 statistical measures
# Depth: 1 channel
#x_repr = tf.reshape( x, [-1, 40, 7, 1] )

# L1 - CONVOLUTIONAL LAYER
# 5x5 patch
# 1 input channel
# 32 features
#W_conv1 = weight_variable([ 5, 5, 1, 32 ])
#b_conv1 = bias_variable([ 32 ])
#h_conv1 = tf.nn.relu( conv2d( x_repr, W_conv1 ) + b_conv1 )
#h_pool1 = max_pool_2x1( h_conv1 )

# L2 - CONVOLUTIONAL LAYER
# 5x5 patch
# 1 input channel
# 64 features
#W_conv2 = weight_variable([ 5, 5, 32, 64 ])
#b_conv2 = bias_variable([ 64 ])
#h_conv2 = tf.nn.relu( conv2d( h_pool1, W_conv2 ) + b_conv2 )
#h_pool2 = max_pool_2x1( h_conv2 )

# L3 - DENSE LAYER
# Input:
# 280 features
# 280 neurons
W_fc1 = weight_variable([ 280, 1024 ])
b_fc1 = bias_variable([ 1024 ])
h_pool2_flat = tf.reshape( x, [-1, 280] )
h_fc1 = tf.nn.relu( tf.matmul( h_pool2_flat, W_fc1 ) + b_fc1 )


# Dropouts
keep_prob  = tf.placeholder( tf.float32 )
h_fc1_drop = tf.nn.dropout( h_fc1, keep_prob )

# LO - SOFTMAX LAYER
W_fc2 = weight_variable([ 1024, 12 ])
b_fc2 = bias_variable([ 12 ])
y = tf.nn.softmax( tf.matmul( h_fc1_drop, W_fc2 ) + b_fc2 )

# Defines the cost function to be minimized.
cost_function = -tf.reduce_mean( y_ * tf.log( y + 1E-10 ) )	# Cross entropy
#cost_function = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))


# Defines the optimization algorithm.
# In this case, the chosen optimization algorithm is the 
# Adam optimizer.
alpha = 0.001
#alpha = 0.0005
train_step = tf.train.AdamOptimizer( alpha ).minimize( cost_function )

# Defines the function that establishes whether a prediction is correct.
# Also, defines the function used for measuring the classifier accuracy.
correct_prediction = tf.equal( tf.argmax( y, 1 ), tf.argmax( y_, 1 ) )
accuracy = tf.reduce_mean( tf.cast( correct_prediction, tf.float32 ) )

# Stores the accuracies of training the models.
accuracies = []

# Iterates through all PAMAP2 subjects.
for subject in range( 1, pamap.NUM_SUBJECTS + 1 ):

	# Loads the LOSO-CV (leave-one-subject-out cross validation for
	# one fold.
	data  = pamap.cross_validation( subject )
	train = data[ 'train' ]
	test  = data[ 'test'  ]

	# Reinitializes all the model variables.
	sess.run( tf.initialize_all_variables() )

	# Computes the mean and variance for each feature.
	# This information will be used for feature normalization.
	means     = np.mean( [ s[ 'features' ] for s in train ], axis = 0 )
	variances = np.var ( [ s[ 'features' ] for s in train ], axis = 0 )

	# Runs the training stage.
	# In each iteration, only a random batch of the training
	# set will be considered for efficiency purposes.
	iters = 20000
	for i in range( iters ):
		batch = pamap.random_sample( train, 0.1 )
		if i%10 == 0:
			train_accuracy = accuracy.eval(feed_dict={
				x  : [ normalize( s[ 'features' ], means, variances ) for s in test ],
				y_ : [ one_hot_encode( 12, s[ 'class'    ] ) for s in test ],
				keep_prob: 1.0
			})
	   		print("step %d, training accuracy %g"%(i, train_accuracy))
			train_step.run( feed_dict = {
				x  : [ normalize( s[ 'features' ], means, variances ) for s in batch ],
				y_ : [ one_hot_encode( 12, s[ 'class'    ] ) for s in batch ],
				keep_prob: 0.5
			})

	# Computes the accuracy of the classifier using the test set.
	model_accuracy = accuracy.eval( feed_dict = {
		x  : [ normalize( s[ 'features' ], means, variances ) for s in test ],
		y_ : [ one_hot_encode( 12, s[ 'class'    ] ) for s in test ],
		keep_prob: 1.0
	})

	accuracies.append( model_accuracy )

print accuracies
print np.mean( accuracies )