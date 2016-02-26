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
	PROCESSED_FILTERED = 2

	FILTER_GA = "01010101000101110001010000000000101101001001011110011100001001100010000010011001000001101000100000111110" \
		   "01000101110110101000001110111100111100100001011000001111110101101111100010001110000001101111000101100001" \
		   "101010000000110011110000100011110010001001010110100111010010110010000001"

	FILTER_MTS = "1101010101110010110100001001111101111001111101111000000001000000101000111101100010000001000010001110001" \
			"0111011001001011111001100110001010101110000000111001110001000100101101000101000000100110010110110101011" \
			"00100000101011111111110000010111011000101111001010010001101011001010000010"

	NO_FILTER = "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111" \
			"1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111" \
			"11111111111111111111111111111111111111111111111111111111111111111111111111"

	NUM_SUBJECTS = 2
	CLASSES = {
		1 : 0,		2 : 1,		3 : 2,		4 : 3,		5 : 4,		6 : 5,
		7 : 6,		12 : 7,		13 : 8,		16 : 9,		17 : 10,	24 : 11	
	}



	LABELS = ['lying', 'sitting', 'standing', 'walking', 'running', 'cycling', 
			  'nordic walking'  , 'ascending stairs'   , 'descending stairs' ,
			  'vacuum cleaning' , 'ironing'            , 'rope jumping'       ]

	def __init__ ( self, source, binary_filter=None ):
		self.data = []
		self.features = 0
		if ( source == self.RAW ):
			self.load_raw()
		elif ( source == self.PROCESSED ):
			self.load_processed()
		elif ( source == self.PROCESSED_FILTERED ):
			self.load_processed_filtered(binary_filter)
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
		self.features = 280

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
		self.features = 280

	def load_processed_filtered ( self , binary_filter):
		# This method only loads some columns (feature selection)
		# param filter is a string {01000101} where 0 is feature not selected and 1 selected
		column_list = self.get_column_list(binary_filter)
		base_path = os.path.dirname( __file__ )
		data_path = os.path.join( base_path, 'data', 'pamap', 'processed' )
		for subject in range( self.NUM_SUBJECTS ):
			filename = os.path.join( data_path, 'subject10' + str(subject + 1) + '.dat' )
			with open(filename) as f:
				data = np.genfromtxt( self.filter_transitions( f, -1 ), delimiter = ' ' , usecols = tuple(column_list) )

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

	def get_column_list ( self, binary_filter ):
		# Transform 010101.. to column number list zero based (1, 4, 5) => extract 2nd column, 5th and 6th
		selected_columns = []
		n_sel = 0
		n_col = 0
		for col in binary_filter:
			if col == '1':
				# include column selected zero based
				selected_columns.append(n_col)
				n_sel += 1
			n_col += 1

		self.features = n_sel
		selected_columns.append(n_col) #Always add the class
		return selected_columns
