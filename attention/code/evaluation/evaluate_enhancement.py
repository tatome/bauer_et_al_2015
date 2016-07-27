import numpy
import bz2
import csv

import argparse
import yaml

import itertools

import collections

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from activation_functions import activation_functions

parser = argparse.ArgumentParser()
parser.add_argument('-c', dest='configfile', required=True)
parser.add_argument('-e', dest='evaluation_configfile', required=True)
parser.add_argument('-m', dest='mapping', required=True)
parser.add_argument('-i', dest='infile', required=True)
parser.add_argument('-o', dest='outfile')
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

real_class = 1
fake_class = 2
def mean_activity_when_dist_max(data, maxdist, discriminator, mapping, which_class):
    numneurons = mapping.shape[0]

    stimulus_class = data[:,which_class]
    relevant = discriminator(stimulus_class)
    relevant_data = data[relevant]

    relevant_activations = relevant_data[:,-numneurons:]
    relevant_truths = relevant_data[:,0]
    dists = numpy.tile(relevant_truths, (numneurons,1)).T - mapping
    selection = (numpy.abs(dists) <= maxdist)
    selected_activations = (relevant_activations * selection)

    indices = numpy.arange(selected_activations.shape[0])
    numpy.random.shuffle(indices)

    num_selected = selection.sum(axis=0)
    mean = selected_activations.sum(axis=0) / num_selected
    return mean, num_selected

def enhancement(data, maxdist, discriminator, discriminator_baseline, mapping, which_class):
    mean_activations, counts = mean_activity_when_dist_max(data, maxdist, discriminator, mapping, which_class)
    assert (counts > 150).all(), collections.Counter(counts)

    mean_activations_baseline, counts_baseline = mean_activity_when_dist_max(data, maxdist, discriminator_baseline, mapping, which_class)
    assert (counts_baseline > 150).all(), collections.Counter(counts_baseline)

    return mean_activations / mean_activations_baseline

maxdist = evaluation_config['feature_enhancement']['max_dist']

visual_params, auditory_params = [(c['activation']['sigma'], c['dim'][0], c['activation']['amplitude'], c['activation']['baseline']) for c in config['modalities']]

position_config = config['position']
position_params = [position_config[key] for key in ['sigmoid_steepness', 'middle_width', 'sigmoid_exc', 'sigmoid_scale', 'sigmoid_baseline']]

class_config = config['class']
class_params = [class_config[key] for key in [ 'class_scale', 'class_baseline']]

activation_functions = activation_functions(visual_params, auditory_params, position_params, class_params)

cog_auditory_enhancement  = enhancement(data, maxdist, activation_functions.is_noisy, activation_functions.is_none, mapping, fake_class)
cog_visual_enhancement    = enhancement(data, maxdist, activation_functions.is_visible, activation_functions.is_none, mapping, fake_class)
cog_both_enhancement      = enhancement(data, maxdist, activation_functions.is_noisy_and_visible, activation_functions.is_none, mapping, fake_class)

sens_auditory_enhancement = enhancement(data, maxdist, activation_functions.is_noisy, activation_functions.is_none, mapping, real_class)
sens_visual_enhancement   = enhancement(data, maxdist, activation_functions.is_visible, activation_functions.is_none, mapping, real_class)
sens_both_enhancement     = enhancement(data, maxdist, activation_functions.is_noisy_and_visible, activation_functions.is_none, mapping, real_class)

numeric_data = {}

# determine where enhancement due to cognitive input is strongest, weakest.
vis_max_enhancement_idx = numpy.argmax(cog_visual_enhancement)
vis_min_enhancement_idx = numpy.argmin(cog_visual_enhancement)
aud_max_enhancement_idx = numpy.argmax(cog_auditory_enhancement)
aud_min_enhancement_idx = numpy.argmin(cog_auditory_enhancement)
both_max_enhancement_idx = numpy.argmax(cog_both_enhancement)
both_min_enhancement_idx = numpy.argmin(cog_both_enhancement)

# compute enhancement due to sensory input for all neurons.
sens_auditory_enhancement = enhancement(data, maxdist, activation_functions.is_noisy, activation_functions.is_none, mapping, real_class)
sens_visual_enhancement   = enhancement(data, maxdist, activation_functions.is_visible, activation_functions.is_none, mapping, real_class)
sens_both_enhancement     = enhancement(data, maxdist, activation_functions.is_noisy_and_visible, activation_functions.is_none, mapping, real_class)

setce = {}
numeric_data['sensory_enhancement_to_cognitive_enhancement'] = setce
setce['visual'] = {
    'vis'  : float(sens_visual_enhancement[vis_max_enhancement_idx]),
    'aud'  : float(sens_auditory_enhancement[vis_max_enhancement_idx]),
    'both' : float(sens_both_enhancement[vis_max_enhancement_idx]),
    'cog_enhancement' : map(float, cog_visual_enhancement),
    'sens_enhancement' : map(float, sens_visual_enhancement)
}
setce['auditory'] = {
    'vis'  : float(sens_visual_enhancement[aud_max_enhancement_idx]),
    'aud'  : float(sens_auditory_enhancement[aud_max_enhancement_idx]),
    'both' : float(sens_both_enhancement[aud_max_enhancement_idx]),
    'cog_enhancement' : map(float, cog_auditory_enhancement),
    'sens_enhancement' : map(float, sens_auditory_enhancement)
}
setce['both'] = {
    'vis'  : float(sens_visual_enhancement[both_max_enhancement_idx]),
    'aud'  : float(sens_auditory_enhancement[both_max_enhancement_idx]),
    'both' : float(sens_both_enhancement[both_max_enhancement_idx]),
    'cog_enhancement' : map(float, cog_both_enhancement),
    'sens_enhancement' : map(float, sens_both_enhancement)
}

strong_enhancement = evaluation_config['stimulus_selectivity']['strong_enhancement']
weak_depression = evaluation_config['stimulus_selectivity']['weak_depression']
setce['num_nonselective'] = int(((sens_visual_enhancement >= strong_enhancement) & (sens_auditory_enhancement >= strong_enhancement) & (sens_both_enhancement > strong_enhancement)).sum())

def discriminating_indifferent(first, second):
    return int(((first >= strong_enhancement) & (second < strong_enhancement) & (second > weak_depression)).sum())

vis_aud_indifferent = discriminating_indifferent(sens_visual_enhancement, sens_auditory_enhancement)
aud_vis_indifferent = discriminating_indifferent(sens_auditory_enhancement, sens_visual_enhancement)

setce['num_vis_aud_indifferent'] = vis_aud_indifferent
setce['num_aud_vis_indifferent'] = aud_vis_indifferent

with open(args.outfile, 'wc') as outfile:
    yaml.dump(data=numeric_data, stream=outfile)

