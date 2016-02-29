# -*- coding: utf-8 -*-
from pamap import PAMAP
import tensorflow as tf
import numpy as np

# Defines the window size.
W = 256

# Converts a scalar to a one-hot encoding vector.
def one_hot_encode ( range, value ):
	vec = np.zeros( range )
	vec[ value ] = 1
	return vec

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

# Initializes a pooling operator.
def max_pool( d1, d2, x ):
  return tf.nn.max_pool( x, ksize=[1, d1, d2, 1], strides=[1, d1, d2, 1], padding='SAME' )

# Loads the preprocessed and normalized PAMAP2 dataset.
pamap = PAMAP( PAMAP.RAW )

# Starts the TensorFlow interactive session.
sess = tf.InteractiveSession()

# Placeholders will be filled with data from the PAMAP dataset.
# Shapes specify the dimensions of data inputs and outputs.
x  = tf.placeholder( tf.float32, shape=[ None, W, 40 ] )		# Number of features: W * 40
y_ = tf.placeholder( tf.float32, shape=[ None, 12  ] )  		# Number of classes: 12

# Reshapes the input data to a more accurate representation
# of the original data structure.
# Width: 40 sensors
# Height: 7 statistical measures
# Depth: 1 channel
x_repr = tf.reshape( x, [-1, W, 40, 1] )

# L1 - CONVOLUTIONAL LAYER
# 64x8 patch
# 1 input channel
# 1024 features
W_conv1 = weight_variable([ W/8, 8, 1, W*2 ])
b_conv1 = bias_variable([ W*2 ])
h_conv1 = tf.nn.relu( conv2d( x_repr, W_conv1 ) + b_conv1 )
h_pool1 = max_pool( W/8, 10, h_conv1 )

# L2 - CONVOLUTIONAL LAYER
# 4x2 patch
# 1 input channel
# 4096 features
W_conv2 = weight_variable([ W/128, 2, W*2, W*4 ])
b_conv2 = bias_variable([ W*4 ])
h_conv2 = tf.nn.relu( conv2d( h_pool1, W_conv2 ) + b_conv2 )
h_pool2 = max_pool( 2, 1, h_conv2 )

# L3 - DENSE LAYER
# 4096 features
# W * 40 neurons
W_fc1 = weight_variable([ 4 * 4 * W * 4, W * 40 ])
b_fc1 = bias_variable([ W * 40 ])
h_pool2_flat = tf.reshape( h_pool2, [-1, 4 * 4 * W * 4] )
h_fc1 = tf.nn.relu( tf.matmul( h_pool2_flat, W_fc1 ) + b_fc1 )

# Dropouts
keep_prob  = tf.placeholder( tf.float32 )
h_fc1_drop = tf.nn.dropout( h_fc1, keep_prob )

# LO - SOFTMAX LAYER
W_fc2 = weight_variable([ W * 40, 12])
b_fc2 = bias_variable([ 12 ])
y = tf.nn.softmax( tf.matmul( h_fc1_drop, W_fc2 ) + b_fc2 )

# Defines the cost function to be minimized.
cost_function = -tf.reduce_mean( y_ * tf.log( y + 1E-10 ) )	# Cross entropy

# Defines the optimization algorithm.
# In this case, the chosen optimization algorithm is the 
# Adam optimizer.
#alpha = 0.001
alpha = 0.01
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

	# Runs the training stage.
	# In each iteration, only a random batch of the training
	# set and the test set will be considered for efficiency purposes.
	iters = 10000
        test_batch  = pamap.random_window_batch( test , W, 100 )
	for i in range( iters ):
		train_batch = pamap.random_window_batch( train, W, 300 )
		train_step.run( feed_dict = {
			x  : [[ s[ 'features' ] for s in w ] for w in train_batch ],
			y_ : [ one_hot_encode( 12, w[ 0 ][ 'class' ] ) for w in train_batch ],
			keep_prob: 0.5
		})
		print sess.run(W_fc1)
		print sess.run(b_fc1)
		print tf.argmax( y, 1 ).eval( feed_dict={
                        x  : [[ s[ 'features' ] for s in w ] for w in test_batch ],
			keep_prob: 1.0 
		})
		test_accuracy = accuracy.eval( feed_dict={
			x  : [[ s[ 'features' ] for s in w ] for w in test_batch ],
			y_ : [ one_hot_encode( 12, w[ 0 ][ 'class' ] ) for w in test_batch ],
			keep_prob: 1.0
		})
   		print("Subject %d - step %d, training accuracy %g"%(subject, i, test_accuracy))
		
	# Computes the accuracy of the classifier using the test set.
	test_batch  = pamap.random_window_batch( test , W, 0.1 * len( test  ) )
	model_accuracy = accuracy.eval( feed_dict = {
		x  : [[ s[ 'features' ] for s in w ] for w in test_batch ],
		y_ : [ one_hot_encode( 12, w[ 0 ][ 'class' ] ) for w in test_batch ],
		keep_prob: 1.0
	})

	accuracies.append( model_accuracy )

print accuracies
print np.mean( accuracies )
