import bz2
import csv
import os
import itertools
import collections
import numpy

class LoggingPhase(object):
	""" 
		Base class for simulation phases which writes truth, stimulus, and net activation to a 
		bz2-compressed csv file. 
	"""

	def __init__(self, name, outputDirectory, training=False):
		""" 
			name			: name of this phase.  Used for naming the output file.
			outputDirectory : directory (name) to write data to.
			training		: Is this a training phase?  (Or can we parallelize?)
		"""

		self.name = name
		self.outputDirectory = outputDirectory
		self.train = training

	def start(self):
		self.outfile = bz2.BZ2File(os.path.join(self.outputDirectory, self.name + ".csv.bz2"), 'w')
		self.writer = csv.writer(self.outfile)

	def finish(self):
		self.outfile.close()
	
	def evaluate(self, network, truth, stimulus, activity):
		if not isinstance(truth, collections.Iterable):
			truth = (truth,)
		def lineFormat(entry):
			if type(entry) == str:
				return [entry]
			elif isinstance(entry, collections.Iterable):
				return itertools.chain.from_iterable(lineFormat(e) for e in entry)
			else:
				return ["%.5e" % entry]
		line = [truth, stimulus, activity.transpose()[0]]
		self.writer.writerow(list(lineFormat(line)))

def partition_indices(partitioning, total):
    partition_names = partitioning.keys()
    relative_partition_sizes = [partitioning[name] for name in partition_names]
    absolute_partition_sizes = numpy.multiply(relative_partition_sizes, total).astype(int)
    partition_ends = numpy.cumsum(absolute_partition_sizes)
    partition_starts = numpy.hstack(([0], partition_ends[:-1]))

    random_indices = numpy.arange(total)
    numpy.random.shuffle(random_indices)
    partition_indices = dict([(name,random_indices[numpy.arange(start, end)]) for name, start, end in zip(partition_names, partition_starts, partition_ends)])
    return partition_indices

def partition(data, partitioning):
    indices = partition_indices(partitioning, data.shape[0])
    partitions = dict([(name,data[indices]) for name,indices in indices.iteritems()])
    return partitions
