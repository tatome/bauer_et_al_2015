# coding=utf-8
#
#!/usr/bin/python2
#
# Copyright 2013 Johannes Bauer, Universitaet Hamburg
#
# This file is free software.  Do with it whatever you like.
# It comes with no warranty, explicit or implicit, whatsoever.
#
# This python script implements a simple SOM in arbitrary dimensions.
#
# If you find it useful or if you have any questions, do not
# hesitate to contact me at 
#   bauer at informatik dot uni dash hamburg dot de.
#

import numpy
import logging

logger = logging.getLogger(__name__)

class SOM(object):
	def __init__(self, dimensions, datalength):
		""" 

			Parameters:
			dimensions : the dimensions of the SOM (sequence of integers).
			datalength : length of input vectors.
		"""
		self.weights = numpy.random.rand(*(tuple(dimensions) + (datalength,))) * .1
		self.dimensions = dimensions

	def response(self, datapoint):
		""" Compute the response of the SOM to a given input.

			Parameters:
			datapoint : an input data point.

			Returns:
			activation of the network --- an array with the same dimensionality as the SOM with the squared 
			euclidean distance of each unit from the datapoint.
		"""
		logger.debug('computing SOM response.')
		return numpy.sum(numpy.square(self.weights - datapoint), axis = (len(self.weights.shape)-1))

	def _h(self, sigma, center):
		def f(*vectors):
			normalizers = [max(1,d-1) for d in self.dimensions]
			normed = [v/s for v,s in zip(vectors, normalizers)]
			sqdiff = [(n-(float(c)/s))**2 for n,c,s in zip(normed, center, normalizers)]
			return numpy.exp(-numpy.sum(sqdiff, axis=0) / sigma**2)
		return numpy.fromfunction(f, self.weights.shape)

	def bmuIndex(self, response):
		""" A convenience method for computing the index of the BMU for some response.

			Parameters:
			response : this SOM's response to some input.

			Returns:
			the index of the BMU (a tuple whose length is the number of dimensions of the SOM grid.)
		"""
		maxindex = response.argmin()
		maxindex = numpy.unravel_index(indices = maxindex, dims = response.shape)
		return maxindex	

	def update(self, datapoint, sigma, strength):
		""" Update the SOM units' weights with the given datapoint.

			Parameters:
			datapoint : the datapoint with which to update.
			sigma : the width of the Gaussian neighborhood interaction function h for the update.
			strength : the strength of the update.

			Returns:
			The response to the network *before the update*.  See response()
		"""
		logger.debug("Updating.  Datapoint: %s.", datapoint)
		response = self.response(datapoint)
		maxindex = self.bmuIndex(response)
		logger.debug("BMU index: %s", maxindex)
		logger.debug("BMU weights before: %s", self.weights[maxindex])
		
		h = strength * self._h(sigma, maxindex)
		self.weights = self.weights + (datapoint - self.weights) * h
		logger.debug("BMU weights after: %s", self.weights[maxindex])

		return response

	def computeActivity(self, datapoint):
		ractivity = self.response(datapoint)
		return 1. / (.00001 + ractivity)

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)

	logging.info("Just a second...")

	from matplotlib import pyplot as plt
	def visualize(som):
		im = numpy.zeros((100,100,3))
		im[:,:,0:2] = som.weights
		plt.imshow(im)
		plt.show()

	logging.info("Training an example 100x100-unit SOM with random data.  Please wait.")

	mysom = SOM((100,100), 2)
	steps = 1000
	for width,strength in zip(numpy.linspace(1.14,.01,steps), numpy.linspace(.3,.001, steps)):
		mysom.update((numpy.random.random(),numpy.random.random()), width, strength)

	logging.info("Done. Showing a visualization of the SOM units' weights.")
	visualize(mysom)

