import numpy

import argparse
import yaml

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-m', dest='mapping', required=True)
parser.add_argument('-o', dest='outfile')
parser.add_argument('--figsize', dest='figsize', type=float, nargs=2, default=(1.,1.))
args = parser.parse_args()

mapping = numpy.load(args.mapping)

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

fig = plt.figure(figsize=args.figsize)
plot = fig.add_subplot(111)

plot.plot(mapping, color='#333333')
plot.set_xlabel("neuron grid position")
plot.set_ylabel("mapping")

fig.tight_layout()

if args.outfile:
    plt.savefig(args.outfile)
else:
    plt.show()

