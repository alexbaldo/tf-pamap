# -*- coding: utf-8 -*-
import numpy as np
import os

NUM_SUBJECTS = 8

def load_raw ( ):
	raw_data = []
	base_path = os.path.dirname( __file__ )
	data_path = os.path.join( base_path, 'data', 'pamap', 'raw' )

	for subject in range( NUM_SUBJECTS ):
		filename = os.path.join( data_path, 'subject10' + str(subject + 1) + '.dat' )
		with open(filename) as f:
			data = np.genfromtxt( f, delimiter = ' ', dtype=str )

		# Removes the invalid orientation signals.
		raw_data.append( np.concatenate([data[:,:16], data[:,20:33], data[:,37:50]], axis=1) )

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
# Fills the missing values with the last valid signal value.
for subject in range( NUM_SUBJECTS ):

	# Loads the subject raw data.
	subj_data  = raw_data[ subject ]

	# Stores the preprocessed data.
	preproc_data = []

	# Creates the preprocessed file.
	preproc_file = open_file( subject + 1, 'preprocessed' )

	# Fills the missing values.
	for s in range( 0, len(subj_data) ):
		preproc_sample = []
		if subj_data[ s ][ 1 ] != '0':
			for i in range( 0, len( subj_data[ s ] ) ):
				val = subj_data[ s ][ i ]
				if val == '?':
					prev, next = ('?', '?')
					for j in reversed(range( 0, s )):
						if subj_data[ j ][ i ] != '?':
							prev = subj_data[ j ][ i ]
							break
					for j in range( s, len(subj_data) ):
						if subj_data[ j ][ i ] != '?':
							next = subj_data[ j ][ i ]
							break
					if prev == '?':
						val = next
					elif next == '?':
						val = prev
					else:
						val = (float(prev) + float(next)) / 2
				preproc_sample.append( val )
			preproc_file.write( str( preproc_sample ).replace(',', '').replace('[', '').replace(']', '').replace('\'', '') + '\n' )
		

