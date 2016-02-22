#
# PAMAP.PY
# Provides an interface for interacting with PAMAP2 dataset.
#
__author__ = 'Baldo'

import os
import numpy as np
import random

class PAMAP:

	RAW = 0
	PROCESSED = 1

	NUM_SUBJECTS = 8
	CLASSES = {
		1 : 0,		2 : 1,		3 : 2,		4 : 3,		5 : 4,		6 : 5,
		7 : 6,		12 : 7,		13 : 8,		16 : 9,		17 : 10,	24 : 11	
	}

	def __init__ ( self, source ):
		self.data = { 'features' : [], 'class' : [] }
		if ( source == self.RAW ):
			self.load_raw()
		elif ( source == self.PROCESSED ):
			self.load_processed()
		else:
			raise Exception( 'Invalid data source.' )

	def filter_transitions ( self, f, k ):
		for i, line in enumerate(f):
			if line.split(' ')[k] != '0':
				yield line

	def load_raw ( self ):
		base_path = os.path.dirname( __file__ )
		data_path = os.path.join( base_path, 'data', 'pamap', 'raw' )

		for subject in range( self.NUM_SUBJECTS ):
			filename = os.path.join( data_path, 'subject10' + str(subject + 1) + '.dat' )
			with open(filename) as f:
				data = np.genfromtxt( self.filter_transitions( f, 1 ), delimiter = ' ', missing_values = '?', filling_values = 0 )
				self.data['features'].append( data[:, 2:] )
				self.data['class'].append( map( lambda c : self.CLASSES[c], data[:, 1] ) )

	def load_processed ( self ):
		base_path = os.path.dirname( __file__ )
		data_path = os.path.join( base_path, 'data', 'pamap', 'processed' )
		for subject in range( self.NUM_SUBJECTS ):
			filename = os.path.join( data_path, 'subject10' + str(subject + 1) + '.dat' )
			with open(filename) as f:
				data = np.genfromtxt( self.filter_transitions( f, -1 ), delimiter = ' ' )
				self.data['features'].append( data[:, :-1] )
				self.data['class'].append( map( lambda c : self.CLASSES[c], data[:, -1] ) )


	def cross_validation ( self, fold ):
		return { 
			'train' : { 
				'features' : [ x for l in (self.data['features'][:fold] + self.data['features'][fold+1:]) for x in l ], 
				'class' : [ x for l in (self.data['class'][:fold] + self.data['class'][fold+1:]) for x in l ]
			},
			'test' : { 'features' : self.data['features'][fold], 'class' : self.data['class'][fold] }
		}


	def random_sample ( self, dataset, sample_pct ):
		return random.sample( dataset, int(sample_pct * len( dataset )) )

