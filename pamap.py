# -*- coding: utf-8 -*-
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

	NUM_SUBJECTS = 2
	CLASSES = {
		1 : 0,		2 : 1,		3 : 2,		4 : 3,		5 : 4,		6 : 5,
		7 : 6,		12 : 7,		13 : 8,		16 : 9,		17 : 10,	24 : 11	
	}

	LABELS = ['lying', 'sitting', 'standing', 'walking', 'running', 'cycling', 
			  'nordic walking'  , 'ascending stairs'   , 'descending stairs' ,
			  'vacuum cleaning' , 'ironing'            , 'rope jumping'       ]

	def __init__ ( self, source ):
		self.data = []
		if ( source == self.RAW ):
			self.load_raw()
		elif ( source == self.PROCESSED ):
			self.load_processed()
		else:
			raise Exception( 'Invalid data source.' )

	def load_raw ( self ):
		base_path = os.path.dirname( __file__ )
		data_path = os.path.join( base_path, 'data', 'pamap', 'raw' )

		for subject in range( self.NUM_SUBJECTS ):
			filename = os.path.join( data_path, 'subject10' + str(subject + 1) + '.dat' )
			with open(filename) as f:
				data = np.genfromtxt( f, delimiter = ' ' )
			self.data.append([{ 
				'features' : x[2:], 
				'class' : self.CLASSES[x[1]]
			} for x in data ])

	def load_processed ( self ):
		base_path = os.path.dirname( __file__ )
		data_path = os.path.join( base_path, 'data', 'pamap', 'processed' )
		for subject in range( self.NUM_SUBJECTS ):
			filename = os.path.join( data_path, 'subject10' + str(subject + 1) + '.dat' )
			with open(filename) as f:
				data = np.genfromtxt( self.filter_transitions( f, -1 ), delimiter = ' ' )
				
			self.data.append([{
				'features' : x[:-1], 
				'class' : self.CLASSES[x[-1]] 
			} for x in data ])

	def random_window ( self, dataset, size ):
		# Stores the random window
		window = []

		# Extracts a random starting sample for the window.
		start = np.random.randint( 0, len(dataset) )

		# Determines the window class.
		wclass = dataset[ start ][ 'class' ]

		# Adds samples to the window until it is completely filled.
		# The window must contain samples of only one class. If data
		# from other class is reached, then the process will stop.
		for i in range( start, min( start + size, len(dataset) ) ):
			if dataset[ i ][ 'class' ] == wclass:
				window.append( dataset[ i ] )

		# In case the window is not complete (because another class is reached)
		# Then it is filled from samples previous to the start sample.
		# This should never reach another class, but if it does, then the
		# window will be smaller than the specified size.
		for i in reversed( range( max( 0, start - (size - len(window)) ), start ) ):
			if dataset[ i ][ 'class' ] == wclass:
				window.insert( 0, dataset[ i ] )

		return window

	def cross_validation ( self, fold ):
		return { 
			'train' : [ x for l in ( self.data[:fold-1] + self.data[fold:] ) for x in l ],
			'test'  : self.data[fold-1]
		}

	def random_sample ( self, dataset, sample_pct ):
		return random.sample( dataset, int( sample_pct * len( dataset ) ) )

	def random_window_batch ( self, dataset, window_size, batch_size ):
		batch = []
		while len( batch ) < batch_size:
			batch.append( self.random_window( dataset, window_size ) )

		return batch
