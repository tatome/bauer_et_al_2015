import numpy
import bz2
import csv

import argparse
import yaml

import itertools

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from activation_functions import activation_functions

parser = argparse.ArgumentParser()
parser.add_argument('-c', dest='configfile', required=True)
parser.add_argument('-e', dest='evaluation_configfile', required=True)
parser.add_argument('-m', dest='mapping', required=True)
parser.add_argument('-i', dest='infile', required=True)
parser.add_argument('-s', dest='outfile')
parser.add_argument('--figsize', dest='figsize', type=float, nargs=2, default=(1.,1.))
args = parser.parse_args()

with open(args.configfile) as configfile:
    config = yaml.load(configfile)

with open(args.evaluation_configfile) as configfile:
    evaluation_config = yaml.load(configfile)

numneurons = config['network']['dim'][0]

with bz2.BZ2File(args.infile) as infile:
    reader = csv.reader(infile)
    data = numpy.array([map(float, line) for line in reader])

mapping = numpy.load(args.mapping)

position_class = data[:,1]

visual_params, auditory_params = [(c['activation']['sigma'], c['dim'][0], c['activation']['amplitude'], c['activation']['baseline']) for c in config['modalities']]

position_config = config['position']
position_params = [position_config[key] for key in ['sigmoid_steepness', 'middle_width', 'sigmoid_exc', 'sigmoid_scale', 'sigmoid_baseline']]

class_config = config['class']
class_params = [class_config[key] for key in [ 'class_scale', 'class_baseline']]

activation_functions = activation_functions(visual_params, auditory_params, position_params, class_params)

left = activation_functions.is_left(position_class)
right = activation_functions.is_right(position_class)
middle = activation_functions.is_middle(position_class)
nowhere = numpy.logical_not(left | right | middle)

left = data[left]
right = data[right]
middle = data[middle]
nowhere = data[nowhere]

def mean_when_max_dist(data, maxdist):
    truths = data[:,0]
    dists = numpy.tile(truths, (numneurons,1)).transpose() - mapping
    selection = (numpy.abs(dists) <= maxdist)
    selected_activations = data[:,-numneurons:] * selection
    return selected_activations.sum(axis=0) / selection.sum(axis=0)

maxdist = evaluation_config['spatial_enhancement']['max_dist']
left_enhancement = mean_when_max_dist(left, maxdist) / mean_when_max_dist(nowhere, maxdist)
right_enhancement = mean_when_max_dist(right, maxdist) / mean_when_max_dist(nowhere, maxdist)
middle_enhancement = mean_when_max_dist(middle, maxdist) / mean_when_max_dist(nowhere, maxdist)

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

plots.plot(sorted(mapping), left_enhancement[numpy.argsort(mapping)], color='#aaaaaa', label='$\Leftarrow$')
plots.plot(sorted(mapping), middle_enhancement[numpy.argsort(mapping)], color='#111111', label='$\otimes$')
lines = plots.plot(sorted(mapping), right_enhancement[numpy.argsort(mapping)], color='#666666', label='$\Rightarrow$', linestyle='dashed')
for line in lines:
    line.set_dashes((4,1))
plots.set_xlabel('neurons\' preferred location $l$')
plots.set_ylabel('enhancement')

plots.legend(ncol=3, frameon=False)

fig.tight_layout()

if args.outfile:
    plt.savefig(args.outfile)
else:
    plt.show()
