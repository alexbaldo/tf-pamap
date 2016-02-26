# -*- coding: utf-8 -*-
from pamap import PAMAP
import os
import numpy as np
import re

# Normalizes a vector to have zero mean and unit variance.
#def normalize ( vector, means, variances ):
#	return (vector - means) / variances

# Creates a new file.
def open_file ( subject, settype ):
	base_path = os.path.dirname( __file__ )
	data_path = os.path.join( base_path, 'neoclassifier', 'pamap' )

	filename = os.path.join( data_path, 'subject' + str(subject) + '_' + settype + '.dat' )
	f = open(filename, 'w')
	return f


# Loads the already-processed PAMAP2 dataset.
# In this processed dataset, features are a statistical summary
# resulting from the FFT of a 512-instances sliding window.
pamap = PAMAP( PAMAP.PROCESSED )

# Iterates through all PAMAP2 subjects.
for subject in range( 1, pamap.NUM_SUBJECTS + 1 ):

	# Loads the LOSO-CV (leave-one-subject-out cross validation for
	# one fold.
	data  = pamap.cross_validation( subject )
	train = data[ 'train' ]
	test  = data[ 'test'  ]

	# Computes the mean and variance for each feature.
	# This information will be used for feature normalization.
	# means     = np.mean( [ s[ 'features' ] for s in train ], axis = 0 )
	# variances = np.var ( [ s[ 'features' ] for s in train ], axis = 0 )

	# Normalizes the dataset to 0-mean and unit-variance.
	# Extracts the class labels.
	train_norm_features = [  s[ 'features' ] for s in train ]
	test_norm_features  = [  s[ 'features' ] for s in test  ]
	train_labels 		= [  s[ 'class' ]    for s in train ]
	test_labels 		= [  s[ 'class' ]    for s in test  ]

	# Computes the maximum and minimum values for each feature.
	maxs_train = np.amax( train_norm_features, axis = 0 )
	mins_train = np.amin( train_norm_features, axis = 0 )
	maxs_test  = np.amax( test_norm_features , axis = 0 )
	mins_test  = np.amin( test_norm_features , axis = 0 )

	# Creates and opens the train and test files.
	train_file = open_file( subject, 'train' )
	test_file  = open_file( subject, 'test'  )

	# Writes the headers.
	train_file.write( '@relation pamap-subject' + str(subject) + '-train\n' )
	test_file.write ( '@relation pamap-subject' + str(subject) + '-test\n'  )
	
	for att in range( 0, 280 ):
		train_file.write( '@attribute att' + str(att + 1) + ' real [' + str(mins_train[ att ]) + ', ' + str(maxs_train[ att ]) + ']\n' )
		test_file.write ( '@attribute att' + str(att + 1) + ' real [' + str(mins_test [ att ]) + ', ' + str(maxs_test [ att ]) + ']\n' )

	# TODO Only binary classes are supported so far by NeoClassifier.
	# We'll try it again later.
	train_file.write( '@attribute activity ' + str(range(0, 12)).replace('[', '{').replace(']', '}') + '\n' )
	test_file.write ( '@attribute activity ' + str(range(0, 12)).replace('[', '{').replace(']', '}') + '\n' )
	
	train_file.write( '@inputs ' + str([ "att" + str(att + 1) for att in range(0, 280) ]).replace('[', '').replace(']', '').replace('\'', '') + '\n' )
	test_file.write ( '@inputs ' + str([ "att" + str(att + 1) for att in range(0, 280) ]).replace('[', '').replace(']', '').replace('\'', '') + '\n' )
	train_file.write( '@outputs activity\n' )
	test_file.write ( '@outputs activity\n' )

	train_file.write( '@data\n' )
	test_file.write ( '@data\n' )

	for i in range(0, len(train_norm_features)):
		train_file.write( re.sub(r'^,', '', re.sub(r' +', ',', str(train_norm_features[i]).replace('\n', '').replace('[', '').replace(']', ''))) + ',' + str(train_labels[i]) + '\n' )

	for i in range(0, len(test_norm_features)):
		test_file.write( re.sub(r'^,', '', re.sub(r' +', ',', str(test_norm_features[i]).replace('\n', '').replace('[', '').replace(']', ''))) + ',' + str(test_labels[i]) + '\n' )

