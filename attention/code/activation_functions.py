from numpy import random
import numpy


class activation_functions(object):
    
    # static variables.
    none_class = 0

    left_class = 1
    middle_class = 2
    right_class = 3
    spatial_classes = (left_class, middle_class, right_class, none_class)

    noisy_and_visible_class = 1
    noisy_class = 2
    visible_class = 3

    stimulus_classes = (noisy_and_visible_class, noisy_class, visible_class, none_class)

    def __init__(self, visual_params, auditory_params, position_params, class_params):
        self.visual_params = visual_params
        self.auditory_params = auditory_params
        self.position_params = position_params
        
        self.sigmoid_steepness, self.middle_width, self.sigmoid_exc, self.sigmoid_scale, self.sigmoid_baseline = position_params
        self.class_scale, self.class_baseline = class_params

    def is_none(self, any_class):
        return any_class == self.none_class

    def is_left(self, spatial_class):
        return spatial_class == self.left_class

    def is_right(self, spatial_class):
        return spatial_class == self.right_class

    def is_middle(self, spatial_class):
        return spatial_class == self.middle_class

    def position_class_activation(self, position):
        leftness = 1./(1 + numpy.exp((position + self.sigmoid_exc - .5)*self.sigmoid_steepness))
        rightness = 1./(1 + numpy.exp(-(position - self.sigmoid_exc - .5)*self.sigmoid_steepness))
        middleness = numpy.exp(-(position - .5)**2 / self.middle_width)
        return numpy.array([leftness, middleness, rightness]) * self.sigmoid_scale + self.sigmoid_baseline

    def position_class_activity(self, position):
        activation = self.position_class_activation(position)
        return numpy.random.random(size=activation.shape) < activation

    def is_noisy_and_visible(self, stimulus_class):
        return stimulus_class == self.noisy_and_visible_class

    def is_noisy(self, stimulus_class):
        return stimulus_class == self.noisy_class

    def is_visible(self, stimulus_class):
        return stimulus_class == self.visible_class

    def stimulus_class_activation(self, stimulus_class):
        noisy_and_visible = self.is_noisy_and_visible(stimulus_class)
        noisy = self.is_noisy(stimulus_class)
        visible = self.is_visible(stimulus_class)

        return numpy.array([noisy_and_visible, noisy, visible])
        
    def stimulus_class_activity(self, stimulus_class):
        activation = self.stimulus_class_activation(stimulus_class) * self.class_scale + self.class_baseline
        return numpy.random.random(size=activation.shape) < activation

    def gaussian(self, mean, sigma, length, amplitude, baseline):
        g = numpy.linspace(0., 1., length) - ((mean - .5) * .8 + .5)
        g = numpy.square(g) / -sigma
        g = numpy.exp(g)
        return amplitude * g + baseline

    def sensory_activity(self, position, real_stimulus_class):
        return random.poisson(self.sensory_activation(position, real_stimulus_class))

    def sensory_activation(self, position, real_stimulus_class):
        if isinstance(position, tuple):
            vis_position,aud_position = position
        else:
            vis_position,aud_position = position,position

        sigma, length, amplitude, baseline = self.visual_params
        amplitude = max(self.is_visible(real_stimulus_class) or self.is_noisy_and_visible(real_stimulus_class), .5) * amplitude
        visual_stimulus = self.gaussian(mean = vis_position, sigma=sigma, length=length, amplitude=amplitude, baseline = baseline)

        sigma, length, amplitude, baseline = self.auditory_params
        amplitude = max(self.is_noisy(real_stimulus_class) or self.is_noisy_and_visible(real_stimulus_class), .5) * amplitude
        auditory_stimulus = self.gaussian(mean = aud_position, sigma=sigma, length=length, amplitude=amplitude, baseline = baseline)
        return numpy.hstack((visual_stimulus, auditory_stimulus))
