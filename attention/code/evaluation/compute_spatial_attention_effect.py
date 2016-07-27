import scipy.stats
import numpy as np
import bz2
import yaml
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='infile', required=True)
parser.add_argument('-c', dest='config', required=True)
parser.add_argument('-m', dest='mapping', required=True)
parser.add_argument('-o', dest='outfile', required=True)
args = parser.parse_args()

logger.info('loading config')
with open(args.config) as configfile:
    config = yaml.load(configfile)
numneurons = config['network']['dim'][0]

logger.info('loading mapping.')
mapping = np.load(args.mapping)

logger.info('loading data.')
with bz2.BZ2File(args.infile) as infile:
    data = np.array([np.fromstring(line, sep=',') for line in infile])

logger.info('computing means, testing significance.')
vis = data[:,0]
aud = data[:,1]
left = data[:,3] == 1
right = data[:,3] == 3

localizations = mapping[data[:,-numneurons:].argmax(axis=1)]
rel_localizations = (vis-localizations) / (vis-aud)
switch = aud < .5
attn_on_aud = np.logical_xor(switch, right)
attn_on_vis = np.logical_not(attn_on_aud)

U,p = scipy.stats.mannwhitneyu(rel_localizations[attn_on_vis], rel_localizations[attn_on_aud])
out_data = {
    'attn_on_aud'   : float(rel_localizations[attn_on_aud].mean()),
    'attn_on_vis'   : float(rel_localizations[attn_on_vis].mean()),
    'U'             : float(U),
}
logger.info('Means: %s', out_data)

logger.info('p-value: %e', p)

assert p < .00001, "p-value not << .0001: is %f" % p

with open(args.outfile, 'w') as outfile:
    yaml.dump(stream=outfile, data=out_data)
