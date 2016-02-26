# -*- coding: utf-8 -*-
from pamap import PAMAP
import numpy as np
import matplotlib.pylab as plt

# Loads the preprocessed raw PAMAP2 dataset.
pamap = PAMAP( PAMAP.RAW )

# Iterates through all PAMAP2 subjects.
for subject in range( 1, pamap.NUM_SUBJECTS + 1 ):

	# Splits the dataset in train and test sets.
	data  = pamap.cross_validation( subject )
	train = data[ 'train' ]
	test  = data[ 'test'  ]

	# Generates random windows.
	window = pamap.random_window( train, 128 )
	matrix = np.matrix([ s['features'] for s in window ])

	print matrix

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.set_aspect('equal')
	plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.ocean)
	plt.colorbar()
	plt.show()