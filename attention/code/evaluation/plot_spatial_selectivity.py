import numpy
import pickle
import bz2
import csv

import argparse
import yaml

import itertools

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-c', dest='configfile', required=True)
parser.add_argument('-m', dest='mapping', required=True)
parser.add_argument('-n', dest='network', required=True)
parser.add_argument('-s', dest='outfile')
parser.add_argument('--figsize', dest='figsize', type=float, nargs=2, default=(1.,1.))
args = parser.parse_args()

with open(args.configfile) as configfile:
    config = yaml.load(configfile)

numneurons = config['network']['dim'][0]

with bz2.BZ2File(args.network) as infile:
    network = pickle.load(infile)

mapping = numpy.load(args.mapping)

counts = network.counts
left = counts[:,:,-6,1] / counts[:,:,-6,0]
right = counts[:,:,-5,1] / counts[:,:,-5,0]
center = counts[:,:,-4,1] / counts[:,:,-4,0]

if args.outfile:
    import matplotlib as mpl
    mpl.use("pgf")
    from matplotlib import rc as mprc
    mprc('text', usetex=True)
    mprc('font', family='serif')
    mprc('font', size=8)
    mprc('mathtext', fontset='stix')
    mprc('mathtext', fallback_to_cm=False)
    mprc('pgf', preamble='\\makeatletter\\@ifundefined{propv}{}{}\\@ifundefined{abs}{}{}\\makeatother')

from matplotlib import pyplot as plt

fig,plots = plt.subplots(1, figsize=args.figsize)

plots.plot(sorted(mapping), left[numpy.argsort(mapping)])
plots.plot(sorted(mapping), right[numpy.argsort(mapping)])
plots.plot(sorted(mapping), center[numpy.argsort(mapping)])


fig.tight_layout()

if args.outfile:
    plt.savefig(args.outfile)
else:
    plt.show()
