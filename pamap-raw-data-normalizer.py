# -*- coding: utf-8 -*-
import numpy as np
import os

NUM_SUBJECTS = 8

# Normalizes a vector to be in the range [0-1].
def normalize ( vector, mins, maxs ):
	return (vector - mins) / (maxs - mins)

def load_raw ( ):
	raw_data = []
	base_path = os.path.dirname( __file__ )
	data_path = os.path.join( base_path, 'data', 'pamap', 'raw' )

	for subject in range( NUM_SUBJECTS ):
		filename = os.path.join( data_path, 'subject10' + str(subject + 1) + '.dat' )
		with open(filename) as f:
			data = np.genfromtxt( f, delimiter = ' ' )

		raw_data.append([{ 
				'ts' : x[0],
				'features' : x[2:], 
				'class' : x[1]
			} for x in data ])

	return raw_data

# Creates a new file.
def open_file ( subject, suffix ):
	base_path = os.path.dirname( __file__ )
	data_path = os.path.join( base_path, 'data', 'pamap', 'raw' )

	filename = os.path.join( data_path, 'subject10' + str(subject) + '_' + suffix + '.dat' )
	f = open(filename, 'w')
	return f


# Loads the raw data
raw_data = load_raw()

# Iterates through all PAMAP2 subjects.
for subject in range( NUM_SUBJECTS ):

	# Loads the subject raw data.
	subj_data  = raw_data[ subject ]

	# Creates the normalized file.
	norm_file = open_file( subject + 1, 'subjnorm' )

	# Computes the minimum and maximum value for each feature.
	# This information will be used for feature normalization.
	mins = np.amin( [ s[ 'features' ] for s in subj_data ], axis = 0 )
	maxs = np.amax( [ s[ 'features' ] for s in subj_data ], axis = 0 )

	# Writes the normalized data to the file.
	for s in subj_data:
		norm_s = normalize( s[ 'features' ], mins, maxs )
		ts = str( s[ 'ts' ] )
		c  = str( int(s[ 'class' ]) )
		data = [ts] + [c] + [ str(v) for v in norm_s ] 
		norm_file.write( str( data ).replace(',', '').replace('[', '').replace(']', '').replace('\'', '').replace('\'', '') + '\n' )
