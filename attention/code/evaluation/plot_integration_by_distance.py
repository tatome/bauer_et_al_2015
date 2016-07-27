import numpy as np
import scipy.stats
import bz2
import csv

import argparse
import yaml

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='infile', required=True)
parser.add_argument('-o', dest='outfile', required=True)
parser.add_argument('--figsize', dest='figsize', type=float, nargs=2, default=(1.,1.))
args = parser.parse_args()

logger.info("Reading data.")
data = np.load(args.infile)


fig,plots = plt.subplots(4,  figsize=args.figsize)

def plot_integration(filter_name, plot, title):
    logger.info("Plotting %s.", title)

    if filter_name is not 'all':
        localizations = data['estimates'][data[filter_name]]
        locations = data['stimulus_locations'][data[filter_name]]
    else:
        localizations = data['estimates']
        locations = data['stimulus_locations']

    offsets = locations[:,1] - locations[:,0]
    
    # Cut off where disparity is < .01 --- the relative distance doesn't make much sense there.
    signif_diff = np.abs(offsets) >= 0.01
    locations = locations[signif_diff]
    localizations = localizations[signif_diff]
    offsets = offsets[signif_diff]

    # Compute the relative distance
    rel_localizations = (localizations - locations[:,0]) / offsets

    steps = 25
    bins = 20
    x,y = np.mgrid[0.:bins+1, 0.:steps+1]
    x /= bins
    y /= steps

    im = []

    in_range = (rel_localizations >= 0) & (rel_localizations < 1)
    for o in np.arange(steps, dtype=float) / steps:
        selection = (offsets >= o) & (offsets < o + (1./steps)) & in_range
        binns = (rel_localizations[selection] * bins).astype(int)
        col = np.array([(binns == binn).sum() for binn in np.arange(bins)], dtype=float)
        col /= col.max()
        im.append(col)
    im = np.array(im, dtype=float).T

    # plot histograms
    plot.pcolormesh(y,x,im, cmap="Greys")

    # plot means
    mean_steps = 50
    mean_window_size = 1./20
    xes = np.linspace(0,1,mean_steps+1)[:-1]
    means = np.array(
        [ rel_localizations[(offsets >= o) & (offsets < o + (1./mean_window_size))].mean() 
            for o in xes
        ]
    )
    plot.plot((xes + .5/mean_steps), means, color='black', linewidth=1.2)
    plot.plot((xes + .5/mean_steps), means, color='white', linewidth=0.9)

    tick_locations = np.linspace(0,1,6)
    plot.yaxis.set_ticks(tick_locations)
    tick_labels = ["%.1f" % t for t in tick_locations]
    tick_labels[0] = '$l_v$ '
    tick_labels[-1] = '$l_a$ '
    plot.yaxis.set_ticklabels(tick_labels)

    plot.set_title("attentional: $\\mathit{%s}$" % title)


plot_integration('is_visible', plots[0], 'Va')
plot_integration('is_noisy', plots[1], 'vA')
plot_integration('is_both', plots[2], 'VA')
plot_integration('is_none', plots[3], 'va')

plt.tight_layout()
    
plt.savefig(args.outfile)
