import yaml
import bz2
import pickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', dest='simulation_config', required=True)
parser.add_argument('-e', dest='evaluation_config', required=True)
parser.add_argument('-a', dest='spatial_attention_effect', required=True)
parser.add_argument('-o', dest='outfile', required=True)
args = parser.parse_args()

def defineInform(name,value):
    try:
        if isinstance(value, int) or value.is_integer():
            rep = str(value)
        elif abs(value):
            rep = str(value).rstrip('0')
    except AttributeError:
        rep = value
    outfile.write("\\newcommand{\\%s}{%s}%%\n" % (name, rep))

def defineRounded(name,value,decimals):
    outfile.write(("\\newcommand{\\%s}{%." + str(decimals) + "f}%%\n") % (name, value))

def defineInt(name,value):
    outfile.write("\\newcommand{\\%s}{\\num[scientific-notation=false]{%d}}%%\n" % (name, value))
def defineSci(name,value):
    outfile.write("\\newcommand{\\%s}{\\num[round-precision=4,round-mode=figures,scientific-notation=true]{%e}}%%\n" % (name, value))

def phases(config):
    p = {}
    for phase in config['simulation']['phases']:
        p[phase['name']] = phase
    return p
    

with open(args.outfile,'w') as outfile:

    # Simulation configuration parameters.
    with open(args.simulation_config) as infile:
        simulation_config = yaml.load(infile)

    defineInt('numOutputNeurons', simulation_config['network']['dim'][0])

    phase_config = phases(simulation_config)
    defineInt('numStepsTraining', phase_config['training']['steps'])
    defineInform('minNeighborhoodWidth', phase_config['training']['minInteractionWidth'])

    defineInt('numStepsMapping', phase_config['mapping']['steps'])
    defineInt('numStepsMappingPerNeuron', phase_config['mapping']['steps'] / simulation_config['network']['dim'][0])

    defineInt('numStepsSpatial', phase_config['spatial']['steps'])
    defineInt('numRepsSpatial', phase_config['spatial']['repetitions'])
    defineInt('numStepsFeature', phase_config['feature']['steps'])
    defineInt('numRepsFeature', phase_config['feature']['repetitions'])
    defineInt('numStepsIncongruentSpatial', phase_config['incongruent-spatial']['steps'])
    defineInt('numStepsIncongruentSpatialFull', 2 * 4 * phase_config['incongruent-spatial']['steps'])

    position_config = simulation_config['position']
    defineInt('posSigmoidSteepness', position_config['sigmoid_steepness'])
    defineInform('posMiddleWidth', position_config['middle_width'])
    defineInform('posLeftSigmoidMiddle', .5 - position_config['sigmoid_exc'])
    defineInform('posRightSigmoidMiddle', .5 + position_config['sigmoid_exc'])
    defineInform('posScale', position_config['sigmoid_scale'])
    defineInform('posBaseline', position_config['sigmoid_baseline'])

    class_config = simulation_config['class']
    defineInform('classActivationBaseline', class_config['class_baseline'])
    defineInform('classActivationMax', class_config['class_scale'] + class_config['class_baseline'])

    vis_config = [c for c in simulation_config['modalities'] if c['name'] == 'visual'][0]
    aud_config = [c for c in simulation_config['modalities'] if c['name'] == 'auditory'][0]
    defineInform('inpVisGain', vis_config['activation']['amplitude'])
    defineInform('inpVisWidth', vis_config['activation']['sigma'])
    defineInform('inpAudGain', aud_config['activation']['amplitude'])
    defineInform('inpAudWidth', aud_config['activation']['sigma'])
    assert(vis_config['activation']['baseline'] == aud_config['activation']['baseline'])
    defineInt('inpBaseline', vis_config['activation']['baseline'])
    assert(vis_config['dim'][0] == aud_config['dim'][0])
    defineInt('inpNeuronsPerModality', vis_config['dim'][0])

    # Simulation evaluation parameters
    with open(args.evaluation_config) as infile:
        evaluation_config = yaml.load(infile)

    defineInform('evalSpatialEnhancementMaxDist', evaluation_config['spatial_enhancement']['max_dist'])
    defineInform('evalStimulusSelectionSmallDist', evaluation_config['stimulus_selection'][0]['max'])
    defineInform('evalStimulusSelectionLargeDist', evaluation_config['stimulus_selection'][1]['min'])
    defineInform('evalStrongEnhancement', evaluation_config['stimulus_selectivity']['strong_enhancement'])
    defineInform('evalWeakDepression', evaluation_config['stimulus_selectivity']['weak_depression'])

    defineInform('Vacolor', evaluation_config['colors']['visual'][1:])
    defineInform('vAcolor', evaluation_config['colors']['auditory'][1:])
    defineInform('VAcolor', evaluation_config['colors']['both'][1:])

    with open(args.spatial_attention_effect) as infile:
        spatial_effects = yaml.load(infile)
    
    defineRounded('attentionAttnOnAud', spatial_effects['attn_on_aud'], 3)
    defineRounded('attentionAttnOnVis', spatial_effects['attn_on_vis'], 3)
