import numpy

import argparse
import yaml

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-c', dest='configfile', required=True)
parser.add_argument('-e', dest='eval_configfile', required=True)
parser.add_argument('-d', dest='datafile')
parser.add_argument('-o', dest='outfile')
parser.add_argument('--figsize', dest='figsize', type=float, nargs=2, default=(1.,1.))
args = parser.parse_args()

with open(args.configfile) as configfile:
    config = yaml.load(configfile)

numneurons = config['network']['dim'][0]

with open(args.eval_configfile) as eval_configfile:
    eval_config = yaml.load(eval_configfile)
    colors = eval_config['colors']
    dashes = eval_config['dashes']
    font_size = eval_config['font_size']

with open(args.datafile) as datafile:
    data = yaml.load(datafile)

setce = data['sensory_enhancement_to_cognitive_enhancement']

cog_auditory_enhancement = setce['auditory']['cog_enhancement']
cog_visual_enhancement = setce['visual']['cog_enhancement']
cog_both_enhancement = setce['both']['cog_enhancement']
cog_enhancements = {
    'visual' : cog_visual_enhancement,
    'auditory' : cog_auditory_enhancement,
    'both' : cog_both_enhancement,
}

sens_auditory_enhancement = setce['auditory']['sens_enhancement']
sens_visual_enhancement = setce['visual']['sens_enhancement']
sens_both_enhancement = setce['both']['sens_enhancement']

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

fig,plots = plt.subplots(3, 1, figsize=args.figsize)

def moving_average(x, length):
    y = numpy.copy(x)
    counts = numpy.ones_like(x)
    for i in range(1, length):
        y[i:] += x[:-i]
        counts[i:] += 1
    return y / counts

def plot_some(axes, x, title, order):
    x = numpy.array(x)
    idx = numpy.argsort(x)
    axes.set_title(title)
    for modality in order:
        y = numpy.array(cog_enhancements[modality])[idx]
        y = moving_average(y, 10)
        lines = axes.plot(x[idx], y, color=colors[modality], linestyle='dashed')
        for line in lines:
            print(dashes[modality], modality)
            line.set_dashes(tuple(dashes[modality]))
        axes.set_xlim(0,x.max())

plot_some(plots[0], sens_visual_enhancement, '\\textit{Va}',   ['auditory', 'both', 'visual'])
plot_some(plots[1], sens_auditory_enhancement, '\\textit{vA}', ['visual', 'both', 'auditory'])
plot_some(plots[2], sens_both_enhancement, '\\textit{VA}',   ['visual', 'auditory', 'both'])

fig.tight_layout()

if args.outfile:
    plt.savefig(args.outfile)
else:
    plt.show()
