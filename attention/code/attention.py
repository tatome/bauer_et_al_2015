import math
import sys
import os.path

import collections
import itertools

import numpy
from numpy import random

import bz2
import yaml
import pickle
import csv
import logging
import argparse

from simulation import Simulation
from util import LoggingPhase

from activation_functions import activation_functions

# initialize logging
parser = argparse.ArgumentParser()
parser.add_argument("-o", dest="outputDirectory", required=True)
parser.add_argument("-n", dest="networkDump")
parser.add_argument("-p", dest="phases", nargs='+')
parser.add_argument('-l', dest='logfile')
parser.add_argument("--processes", dest="processes", type=int, default=1)
args = parser.parse_args()
log_format = '%(levelname)s %(name)s %(asctime)s: %(message)s'
logging.basicConfig(filename=args.logfile, format=log_format, level=logging.INFO, filemode = 'w')
logger = logging.getLogger(__name__)
if args.logfile:
    logger.info("Logging to file: %s", args.logfile)

if args.networkDump is not None:
    with bz2.BZ2File(args.networkDump) as dumpfile:
        network = pickle.load(dumpfile)
else:
    network = None

# control randomness
random.seed(42)

# concepts: left, right, middle, noisy, visible, noisy_and_visible
no_concepts = 6

# load config
config = yaml.load(open(os.path.join(args.outputDirectory, 'config.yaml')))
config['network']['inputs'] = sum((c['dim'][0] for c in config['modalities'])) + no_concepts

# simulation-specific code

phaseConfigs = dict([c['name'],c] for c in config['simulation']['phases'])
def phaseConfig(name):
    return phaseConfigs[name]


class TrainingPhase(object):
    """ Phase one of the simulation. Training phase."""
    train = True
    name = 'training'

    def __init__(self):
        self.maxInteractionWidth = phaseConfig('training')['initialInteractionWidth']
        self.minInteractionWidth = phaseConfig('training')['minInteractionWidth']
        self.interactionDecay = phaseConfig('training')['interactionDecay']

        self.steps = phaseConfig('training')['steps']
        self.interactionWidthBase = 1e-5**(1./self.interactionDecay)
        self.trainSom = 'type' in config['network'] and 'som' == config['network']['type']

        visual_params, auditory_params = [(c['activation']['sigma'], c['dim'][0], c['activation']['amplitude'], c['activation']['baseline']) for c in config['modalities']]

        position_config = config['position']
        position_params = [position_config[key] for key in ['sigmoid_steepness', 'middle_width', 'sigmoid_exc', 'sigmoid_scale', 'sigmoid_baseline']]

        class_config = config['class']
        class_params = [class_config[key] for key in [ 'class_scale', 'class_baseline']]
        
        self.activation_functions = activation_functions(visual_params, auditory_params, position_params, class_params)


    def stimulusGenerator(self):

        for step in xrange(self.steps):
            stimulus_class = (self.activation_functions.noisy_and_visible_class, self.activation_functions.noisy_class, self.activation_functions.visible_class)[random.randint(low=0,high=3)]
            position = random.random()

            sensory = self.activation_functions.sensory_activity(position, stimulus_class)

            class_activity = numpy.append(self.activation_functions.position_class_activity(position), self.activation_functions.stimulus_class_activity(stimulus_class))

            stimulus = numpy.hstack((sensory, class_activity))

            truth = (position,stimulus_class)

            yield { 'truth' : truth, 'stimulus' : stimulus }
    
    def interactionWidth(self, step):
        return (self.maxInteractionWidth - self.minInteractionWidth) * self.interactionWidthBase**step + self.minInteractionWidth
    
    def updateStrength(self, step):
        if self.trainSom:
            return .5 - .5 * float(step + 1) / (self.steps + 1)
        else:
            return math.sqrt(step) * 1e-6
    
    def start(self):
        pass

    def finish(self):
        outfile = bz2.BZ2File(os.path.join(args.outputDirectory, "network.pickle.bz2"), 'w')
        pickle.dump(self.network, outfile)
        outfile.close()
    
    def evaluate(self, network, truth, stimulus, activity):
        self.network = network

class MappingPhase(LoggingPhase):
    """ Phase two of the simulation. Mapping phase """
    train = False
    name = 'mapping'

    def __init__(self):
        super(MappingPhase,self).__init__('mapping', args.outputDirectory, False)
        self.steps = phaseConfig('mapping')['steps']

        visual_params, auditory_params = [(c['activation']['sigma'], c['dim'][0], c['activation']['amplitude'], c['activation']['baseline']) for c in config['modalities']]
        position_config = config['position']
        position_params = [position_config[key] for key in ['sigmoid_steepness', 'middle_width', 'sigmoid_exc', 'sigmoid_scale', 'sigmoid_baseline']]

        class_config = config['class']
        class_params = [class_config[key] for key in [ 'class_scale', 'class_baseline']]
        
        self.activation_functions = activation_functions(visual_params, auditory_params, position_params, class_params)

    def stimulusGenerator(self):

        for position in numpy.linspace(0,1,self.steps):
            logging.info("%s: position %f", self.name, position)
            for stimulus_class in self.activation_functions.stimulus_classes:

                sensory = self.activation_functions.sensory_activity(position, stimulus_class)

                spatial_class_activation = [0,0,0]
                class_activation = numpy.append(spatial_class_activation, self.activation_functions.stimulus_class_activation(stimulus_class))
                class_activity = class_activation # this is deterministic --- want to test for all classes.

                stimulus = numpy.hstack((sensory, class_activity))

                truth = (position,stimulus_class)

                yield { 'truth' : truth, 'stimulus' : stimulus }

class SpatialPhase(LoggingPhase):
    """ Test influence of spatial attention. """
    train = False
    name = 'spatial'

    def __init__(self):
        super(SpatialPhase,self).__init__('spatial', args.outputDirectory, False)
        self.steps = phaseConfig('spatial')['steps']
        self.repetitions = phaseConfig('spatial')['repetitions']

        visual_params, auditory_params = [(c['activation']['sigma'], c['dim'][0], c['activation']['amplitude'], c['activation']['baseline']) for c in config['modalities']]
        position_config = config['position']
        position_params = [position_config[key] for key in ['sigmoid_steepness', 'middle_width', 'sigmoid_exc', 'sigmoid_scale', 'sigmoid_baseline']]

        class_config = config['class']
        class_params = [class_config[key] for key in [ 'class_scale', 'class_baseline']]
        
        self.activation_functions = activation_functions(visual_params, auditory_params, position_params, class_params)

    def stimulusGenerator(self):

        for position in numpy.linspace(0,1,self.steps):
            logger.info("spatial: position = %f", position)
            for repetitions in range(self.repetitions):
                for stimulus_class in self.activation_functions.stimulus_classes:
                    sensory = self.activation_functions.sensory_activity(position, stimulus_class)
                    for spatial_class in self.activation_functions.spatial_classes:

                        spatial_class_activity = numpy.array(self.activation_functions.spatial_classes[:-1]) == spatial_class
                        stimulus_class_activity = numpy.array(self.activation_functions.stimulus_classes[:-1]) == stimulus_class

                        class_activity = numpy.hstack((spatial_class_activity, stimulus_class_activity))

                        stimulus = numpy.hstack((sensory, class_activity))

                        truth = (position,spatial_class)

                        yield { 'truth' : truth, 'stimulus' : stimulus }


class FeaturePhase(LoggingPhase):
    """ Test influence of feature attention. """
    train = False
    name = 'feature'

    def __init__(self):
        super(FeaturePhase,self).__init__('feature', args.outputDirectory, False)
        self.steps = phaseConfig('feature')['steps']
        self.repetitions = phaseConfig('feature')['repetitions']

        visual_params, auditory_params = [(c['activation']['sigma'], c['dim'][0], c['activation']['amplitude'], c['activation']['baseline']) for c in config['modalities']]
        position_config = config['position']
        position_params = [position_config[key] for key in ['sigmoid_steepness', 'middle_width', 'sigmoid_exc', 'sigmoid_scale', 'sigmoid_baseline']]

        class_config = config['class']
        class_params = [class_config[key] for key in [ 'class_scale', 'class_baseline']]
        
        self.activation_functions = activation_functions(visual_params, auditory_params, position_params, class_params)

    def stimulusGenerator(self):
        for position in numpy.linspace(0,1,self.steps):
            logger.info("feature: position = %f", position)
            for repetitions in range(self.repetitions):
                for real_stimulus_class in self.activation_functions.stimulus_classes:
                    sensory = self.activation_functions.sensory_activity(position, real_stimulus_class)

                    for fake_stimulus_class in self.activation_functions.stimulus_classes:
                        spatial_class_activity = [0,0,0]
                        stimulus_class_activity = self.activation_functions.stimulus_class_activation(fake_stimulus_class)

                        class_activity = numpy.hstack((spatial_class_activity, stimulus_class_activity))

                        stimulus = numpy.hstack((sensory, class_activity))

                        truth = (position,real_stimulus_class,fake_stimulus_class)

                        yield { 'truth' : truth, 'stimulus' : stimulus }

class IncongruentPhase(LoggingPhase):
    """ incongrunent multi-modal stimuli. """
    train = False
    name = 'incongruent'

    def __init__(self):
        super(IncongruentPhase,self).__init__('incongruent', args.outputDirectory, False)
        self.steps = phaseConfig('incongruent')['steps']

        visual_params, auditory_params = [(c['activation']['sigma'], c['dim'][0], c['activation']['amplitude'], c['activation']['baseline']) for c in config['modalities']]
        position_config = config['position']
        position_params = [position_config[key] for key in ['sigmoid_steepness', 'middle_width', 'sigmoid_exc', 'sigmoid_scale', 'sigmoid_baseline']]

        class_config = config['class']
        class_params = [class_config[key] for key in [ 'class_scale', 'class_baseline']]
        
        self.activation_functions = activation_functions(visual_params, auditory_params, position_params, class_params)

    def stimulusGenerator(self):
        for step in range(self.steps):
            vis_position,aud_position = random.random(size=2)

            sensory = self.activation_functions.sensory_activity((vis_position,aud_position),  self.activation_functions.noisy_and_visible_class)

            for stimulus_class in self.activation_functions.stimulus_classes:
                spatial_class_activity = [0,0,0]
                class_activation = numpy.append(spatial_class_activity, self.activation_functions.stimulus_class_activation(stimulus_class))
                class_activity = class_activation

                stimulus = numpy.hstack((sensory, class_activity))

                truth = (vis_position,aud_position,stimulus_class)

                yield { 'truth' : truth, 'stimulus' : stimulus }

class IncongruentSpatialPhase(LoggingPhase):
    """ incongrunent multi-modal stimuli with spatial attention. """
    train = False
    name = 'incongruent-spatial'

    def __init__(self):
        super(IncongruentSpatialPhase,self).__init__('incongruent-spatial', args.outputDirectory, False)
        self.steps = phaseConfig('incongruent-spatial')['steps']

        visual_params, auditory_params = [(c['activation']['sigma'], c['dim'][0], c['activation']['amplitude'], c['activation']['baseline']) for c in config['modalities']]
        position_config = config['position']
        position_params = [position_config[key] for key in ['sigmoid_steepness', 'middle_width', 'sigmoid_exc', 'sigmoid_scale', 'sigmoid_baseline']]

        class_config = config['class']
        class_params = [class_config[key] for key in [ 'class_scale', 'class_baseline']]
        
        self.activation_functions = activation_functions(visual_params, auditory_params, position_params, class_params)

    def stimulusGenerator(self):
        for step in range(self.steps):
            orig_vis_position = random.random() * 1./3
            orig_aud_position = 1 - orig_vis_position

            for switch in (True,False):
                if switch:
                    vis_position = 1 - orig_vis_position
                    aud_position = 1 - orig_aud_position
                else:
                    vis_position = orig_vis_position
                    aud_position = orig_aud_position

                for spatial_class in (self.activation_functions.left_class, self.activation_functions.right_class):

                    for stimulus_class in self.activation_functions.stimulus_classes:
                        sensory = self.activation_functions.sensory_activity((vis_position,aud_position), stimulus_class)
                        spatial_class_activity = numpy.array(self.activation_functions.spatial_classes[:-1]) == spatial_class

                        class_activation = numpy.append(spatial_class_activity, self.activation_functions.stimulus_class_activity(stimulus_class))
                        class_activity = class_activation

                        stimulus = numpy.hstack((sensory, class_activity))

                        truth = (vis_position,aud_position,stimulus_class,spatial_class)

                        yield { 'truth' : truth, 'stimulus' : stimulus }

phases = {
    'training' : TrainingPhase, 
    'mapping' : MappingPhase, 
    'incongruent' : IncongruentPhase, 
    'spatial' : SpatialPhase, 
    'feature' : FeaturePhase, 
    'incongruent-spatial' : IncongruentSpatialPhase,
}

if args.phases:
    logger.info("Running phases %s", args.phases)
    phases = map(lambda name : phases[name](), args.phases)
else:
    logger.info("Running all phases")
    phases = map(lambda phase : phases[phase['name']](), config['simulation']['phases'])

if args.processes == -1:
    args.processes = None

# run simulation.
simulation = Simulation(config, phases, network, args.processes)
logger.info('starting simulation.')
simulation.run()

logger.info("Done.")

