import time
import math
import numpy
import logging

logger = logging.getLogger(__name__)

minvalue = 1e-150

class Network(object):
    def __init__(self, config):
        self.config = config
        self.net_creation_id = time.time()
        self.net_training_id = None
        self.p = config['network']['p']
        self.con = self.initArray(config['network']['dim'])
        self.coff = self.initArray(config['network']['dim'])
        self.dims = config['network']['dim'][0], config['network']['dim'][1], config['network']['inputs']
        self.counts = self.initArray(self.dims + (1,))
        
        if 'maxexp' in config['network']:
            self.maxexp = config['network']['maxexp']
        else:
            self.maxexp = None

        logger.debug('counts dimensions : %s', self.dims)
    
    def initArray(self, dims):
        a = numpy.empty(dims, dtype=numpy.float64)
        a.fill(minvalue)
        return a
    
    def ensureLength(self, l):
        if self.counts.shape[3] <= l:
            extension = self.initArray(self.dims + (l - self.counts.shape[3] + 2,))
            self.counts = numpy.append(self.counts, extension, 3)
            logger.debug("Extended counts array to size %s.", self.counts.shape)
        
    def update(self, inputActivity, sigma, s):
        logger.debug('updating connections.\n\tsigma: %f, s: %f', sigma, s)
        activity = self.computeActivity(inputActivity)
        maxindex = self.bmuIndex(activity)

        def deltasq(k,b):
            return (float(k[0]-b[0])/ self.dims[0])**2 + (float(k[1]-b[1])/ self.dims[1])**2
        
        def h(deltasq, sigma):
            """ Update strength.  Depends on the Euclidean distance between x,y and maxindex."""
            return math.exp(-deltasq / sigma)

        for x in xrange(self.dims[0]):
            for y in xrange(self.dims[1]):
                eta = h(deltasq((x,y), maxindex), sigma)
                u = s * eta
                self.coff[x,y] += s
                self.con[x,y] += u
                for i in xrange(self.dims[2]):
                    self.counts[x,y,i,inputActivity[i]] += u

        return activity

    def bmuIndex(self, activity):
        maxindex = activity.argmax()
        maxindex = numpy.unravel_index(indices = maxindex, dims = activity.shape)
        return maxindex
    
    def computeActivity(self, inputActivity):
        logger.debug('computing activity.')
        self.ensureLength(inputActivity.max())

        # numpy array magic
        idx = numpy.mgrid[0:self.dims[0], 0:self.dims[1], 0:self.dims[2]]
        tInputActivity = numpy.tile(inputActivity, self.dims[:-1] + (1,))
        factors = 2 * self.counts[idx[0],idx[1],idx[2],tInputActivity] / numpy.sum(self.counts, axis=3)
        mans,exps = numpy.frexp(factors)
        mantissas, exponents = numpy.frexp(numpy.prod(mans, axis=2))
        exponents += exps.sum(axis=2)

        if self.maxexp is not None:
            maxexp = self.maxexp
        else:
            maxexp = exponents.max()

        exponents -= maxexp
        logger.debug("Maximum exponent: %d", maxexp)
        activity = mantissas * numpy.exp2(exponents)

        if self.p != 0:
            conscience = (self.coff / self.con)**self.p
            activity *= conscience

        activity *= numpy.prod(activity.shape) / activity.sum()
        return activity
