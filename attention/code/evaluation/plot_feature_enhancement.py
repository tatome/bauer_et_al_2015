import argparse
import yaml

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-c', dest='configfile', required=True)
parser.add_argument('-e', dest='evaluation_configfile', required=True)
parser.add_argument('-d', dest='datafile')
parser.add_argument('-o', dest='outfile')
parser.add_argument('--figsize', dest='figsize', type=float, nargs=2, default=(1.,1.))
args = parser.parse_args()

with open(args.configfile) as configfile:
    config = yaml.load(configfile)

numneurons = config['network']['dim'][0]

with open(args.datafile) as datafile:
    data = yaml.load(datafile)

with open(args.evaluation_configfile) as eval_configfile:
    eval_config = yaml.load(eval_configfile)
    colors = eval_config['colors']
    dashes = eval_config['dashes']
    font_size = eval_config['font_size']

setce = data['sensory_enhancement_to_cognitive_enhancement']
auditory_enhancement = setce['auditory']['cog_enhancement']
visual_enhancement = setce['visual']['cog_enhancement']
both_enhancement = setce['both']['cog_enhancement']

sens_auditory_enhancement = setce['auditory']['sens_enhancement']
sens_visual_enhancement = setce['visual']['sens_enhancement']
sens_both_enhancement = setce['both']['sens_enhancement']

if args.outfile:
    import matplotlib as mpl
    mpl.use("pgf")
    from matplotlib import rc as mprc
    mprc('text', usetex=True)
    mprc('font', family='serif')
    mprc('font', size=font_size)
    mprc('mathtext', fontset='stix')
    mprc('mathtext', fallback_to_cm=False)
    mprc('pgf', preamble='\\makeatletter\\@ifundefined{propv}{}{}\\@ifundefined{abs}{}{}\\makeatother')

from matplotlib import pyplot as plt

fig,plots = plt.subplots(2, figsize=args.figsize)
def plot_em(plot, curves, color, dashes, label):
    lines = plot.plot(range(numneurons), curves, color=color, linestyle='dashed', label=label)
    for line in lines:
        line.set_dashes(tuple(dashes))

plots[0].set_title("Enhancement due to Class of Attentional Input")
plot_em(plots[0], visual_enhancement, colors['visual'], dashes['visual'], '$aV$')
plot_em(plots[0], auditory_enhancement, colors['auditory'], dashes['auditory'], '$Av$')
plot_em(plots[0], both_enhancement, colors['both'], dashes['both'], '$AV$')

plots[1].set_title("Enhancement due to Class of Sensory Input")
plot_em(plots[1], sens_visual_enhancement, colors['visual'], dashes['visual'], '$aV$')
plot_em(plots[1], sens_auditory_enhancement, colors['auditory'], dashes['auditory'], '$Av$')
plot_em(plots[1], sens_both_enhancement, colors['both'], dashes['both'], '$AV$')

fig.tight_layout()

if args.outfile:
    plt.savefig(args.outfile)
else:
    plt.show()
