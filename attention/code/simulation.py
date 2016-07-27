import sys
import logging
import itertools
import time
import threading
from multiprocessing import Pool
from network import Network
from som import SOM

logger = logging.getLogger(__name__)

class Simulation(object):
    def __init__(self, config, phases, network = None, processes=1):
        self.phases = phases
        if network is None:
            if 'type' in config['network'] and config['network']['type'] == 'som':
                logger.info('initializing regular SOM.')
                dim = config['network']['dim']
                datalength = sum([m['dim'][0] * m['dim'][1] for m in config['modalities']])
                self.network = SOM(dimensions = dim, datalength = datalength)
            else:
                logger.info('initializing som variant.')
                self.network = Network(config)
        else:
            self.network = network
        self.processes = processes
    

    def run(self):
        for phase in self.phases:
            logger.info('Starting phase: %s', phase.name)
            phase.start()
            if phase.train:
                for step,stimulus in enumerate(phase.stimulusGenerator()):
                    sigma = phase.interactionWidth(step)
                    s = phase.updateStrength(step)

                    logger.debug('Phase: %s, Step: %d. Sigma: %f, Strength: %f', phase.name, step, sigma, s)
                    if not logger.isEnabledFor(logging.DEBUG) and step % 100 == 0:
                        logger.info('Phase: %s, Step: %d. Sigma: %f, Strength: %f', phase.name, step, sigma, s)

                    activity = self.network.update(stimulus['stimulus'], sigma, s)
                self.network.net_training_id = time.time()
                phase.evaluate(self.network, stimulus['truth'], stimulus['stimulus'], activity)

            else:
                pid = (threading.current_thread().ident,time.time())
                if self.processes is not None:
                    logger.info("Using %d processes.", self.processes)
                else:
                    logger.info("Using as many processes as deemed appropriate by Python.")

                pools[pid] = Pool(self.processes)
                p = pools[pid]

                def repeater():
                    for _step, stimulus in enumerate(phase.stimulusGenerator()):
                        yield pid, self.network, _step, stimulus

                for step, stimulus, activity in p.imap(_doStep, repeater(), chunksize=100):
                    if step % 100 == 0:
                        logger.info('Phase: %s, Step: %d', phase.name, step)
                    phase.evaluate(self.network, stimulus['truth'], stimulus['stimulus'], activity)

                pools[pid] = None

            phase.finish()

pools = {}
def _doStep((pid, network, step, stimulus)):
    try:
        activity = network.computeActivity(stimulus['stimulus'])
        return step, stimulus, activity
    except Exception as e1:
        if pid in pools:
            logger.error(e1)
            logger.info("Terminating.")
            sys.exit()
        raise
