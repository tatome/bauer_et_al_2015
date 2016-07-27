import argparse
import yaml
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='infile', required=True)
parser.add_argument('-o', dest='outfile')
parser.add_argument('--figsize', dest='figsize', type=float, nargs=2, default=(1.,1.))
args = parser.parse_args()

with open(args.infile) as infile:
    data = yaml.load(infile)
    data = data['sensory_enhancement_to_cognitive_enhancement']

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

fig,plots = plt.subplots(1, 3, figsize=args.figsize)

def plot_bars(plot, modality, label):
    height = [data[modality][key] for key in ('vis', 'aud', 'both')]
    plot.set_ylim(-.05 * max(height), 1.05 * max(height))
    plot.bar(left=[1,2,3], height=height, color='gray')
    plot.set_title(label)
    plot.set_xticks([1.4,2.4,3.4])
    plot.set_xticklabels(['$aV$', '$Av$', '$AV$'])
    plot.set_xlim(.5,4.3)

plot_bars(plots[0], 'visual', 'most visual')
plot_bars(plots[1], 'auditory', 'most auditory')
plot_bars(plots[2], 'both', 'most both')
plots[1].set_xlabel('(sensory) stimulus class')

plots[0].set_ylabel('enhancement')

fig.tight_layout()

if args.outfile:
    plt.savefig(args.outfile)
else:
    plt.show()
