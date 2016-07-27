import numpy

import argparse
import yaml

import itertools

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from activation_functions import activation_functions

parser = argparse.ArgumentParser()
parser.add_argument('-c', dest='configfile', required=True)
parser.add_argument('-o', dest='outfile')
parser.add_argument('--figsize', dest='figsize', type=float, nargs=2, default=(1.,1.))
args = parser.parse_args()

x = numpy.linspace(0,1,500)

with open(args.configfile) as configfile:
    config = yaml.load(configfile)

visual_params, auditory_params = [(c['activation']['sigma'], c['dim'][0], c['activation']['amplitude'], c['activation']['baseline']) for c in config['modalities']]

position_config = config['position']
position_params = [position_config[key] for key in ['sigmoid_steepness', 'middle_width', 'sigmoid_exc', 'sigmoid_scale', 'sigmoid_baseline']]

class_config = config['class']
class_params = [class_config[key] for key in [ 'class_scale', 'class_baseline']]

activation_functions = activation_functions(visual_params, auditory_params, position_params, class_params)

activation = activation_functions.position_class_activation(x)

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

plots.plot(x, activation[0], label='$\Leftarrow$', color='#aaaaaa')
plots.plot(x, activation[1], label='$\otimes$', color='#111111')
lines = plots.plot(x, activation[2], label='$\Rightarrow$', color='#666666', linestyle='dashed')
for line in lines:
    line.set_dashes((4,1))
plots.set_ylabel('activation $\hat{\mathbf{a}}_{p}$')
plots.set_xlabel('position $l$')
plots.legend(ncol=3, frameon=False)

fig.tight_layout()

if args.outfile:
    plt.savefig(args.outfile)
else:
    plt.show()
