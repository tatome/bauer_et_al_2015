import numpy as np
import bz2
import csv

import argparse
import yaml

import logging


from activation_functions import activation_functions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-c', dest='configfile', required=True)
parser.add_argument('-m', dest='mapping', required=True)
parser.add_argument('-i', dest='infile', required=True)
parser.add_argument('-o', dest='outfile', required=True)
args = parser.parse_args()

with open(args.configfile) as configfile:
    config = yaml.load(configfile)
numneurons = config['network']['dim'][0]

logger.info("Reading data.")
with bz2.BZ2File(args.infile) as infile:
    data = np.array([np.fromstring(line, sep=',') for line in infile])

mapping = np.load(args.mapping)

stimulus_locations = data[:,0:2]
stimulus_class = data[:,2]
estimates = mapping[data[:,-numneurons:].argmax(axis=1)]

# compute stimulus classes.
visual_params, auditory_params = [(c['activation']['sigma'], c['dim'][0], c['activation']['amplitude'], c['activation']['baseline']) for c in config['modalities']]

position_config = config['position']
position_params = [position_config[key] for key in ['sigmoid_steepness', 'middle_width', 'sigmoid_exc', 'sigmoid_scale', 'sigmoid_baseline']]

class_config = config['class']
class_params = [class_config[key] for key in [ 'class_scale', 'class_baseline']]

activation_functions = activation_functions(visual_params, auditory_params, position_params, class_params)

visible = activation_functions.is_visible(data[:,2])
noisy = activation_functions.is_noisy(data[:,2])
both = activation_functions.is_noisy_and_visible(data[:,2])
none = np.logical_not(visible | noisy | both)

out_data = {
        'stimulus_locations' : stimulus_locations,
        'estimates' : estimates,
        'is_visible' : visible,
        'is_noisy' : noisy,
        'is_both' : both,
        'is_none' : none,
        'config' : config,
}

np.savez(file=args.outfile, **out_data)
