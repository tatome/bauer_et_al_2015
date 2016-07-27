import numpy as np
import scipy.stats
import bz2
import csv

import argparse
import yaml

import logging

import myem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='infile', required=True)
parser.add_argument('-o', dest='outfile', required=True)
args = parser.parse_args()

logger.info("Reading data.")
data = np.load(args.infile)

visible = data['is_visible']
noisy = data['is_noisy']
both = data['is_both']
none = data['is_none']

# currently only have numpy 1.6 --- no inbuilt nanmean.
def nanmean(d):
    return np.nansum(d) / np.isfinite(d).sum()

# currently only have numpy 1.6 --- no inbuilt nanmean.
def nanstd(d):
    return np.sqrt((d-nanmean(d))**2/np.isfinite(d).sum())    

def aicc(log_likelihood, k, n):
    # k : number of parameters
    # n : sample size
    aic = 2 * k - 2 * log_likelihood
    return aic + 2 * k * (k+1) / (n - k - 1)

def akaike_integration(data):
    # fit Gaussian
    mu = data.mean()
    sigma = data.std()
    likelihoods = 1./np.sqrt(2 * np.pi * sigma**2) * np.exp(-(mu-data)**2/(2*sigma**2))
    log_likelihoods = np.log(likelihoods)
    log_likelihood = np.sum(log_likelihoods)
    return aicc(log_likelihood, 2, len(data))

def akaike_selection(data):
    sigma_one, sigma_two, a = myem.fit(data, 10000, 1e-10)
    log_likelihood = myem.log_likelihood(data, sigma_one, sigma_two, a)
    return aicc(log_likelihood, 3, len(data)), a

def compute_localization_statistics(filter_name):
    logger.info('Computing statistics for filter "%s"', filter_name)
    if filter_name is not 'all':
        localizations = data['estimates'][data[filter_name]]
        locations = data['stimulus_locations'][data[filter_name]]
    else:
        localizations = data['estimates']
        locations = data['stimulus_locations']

    offsets = locations[:,1] - locations[:,0]
    
    signif_diff = np.abs(offsets) >= 0.01

    # Cut off where disparity is < .01 --- the relative distance doesn't make much sense there.
    locations = locations[signif_diff]
    localizations = localizations[signif_diff]
    offsets = offsets[signif_diff]

    # Compute the relative distance
    rel_localizations = (localizations - locations[:,0]) / offsets

    logger.debug("Slicing data by offset.")
    # slice by offset
    samples = {}
    soffsets = {}
    window_size = 1./20
    for o in np.linspace(0,1,500):
        selection  = (np.abs(offsets - o) < window_size)
        samples[o] = rel_localizations[selection]
        soffsets[o] = offsets[selection]
    dist  = np.sort(samples.keys())

    logger.debug("Computing means")
    means = np.array([nanmean(samples[step]) for step in dist])

    logger.debug("Computing standard deviations")
    # what we need is relative stds of independent size.
    stds = np.array([np.sqrt(np.nansum(((samples[step] - nanmean(samples[step])) * offsets[step])**2)/np.isfinite(samples[step]).sum()) for step in dist])

    # compute akaike information criterion
    # AICC for stimulus integration
    logger.debug("Computing AICCs for stimulus integration model")
    aicc_int = np.array([akaike_integration(samples[step]) for step in dist])
    logger.debug("Computing AICCs for stimulus selection model")
    aicc_sel,ratios = np.array([akaike_selection(samples[step]) for step in dist]).T

    return np.vstack((dist, means, stds, aicc_int, aicc_sel, ratios))

def global_mean(filter_name):
    # filtering again, same way as above---wasteful, but clear.
    logger.info('Computing global mean for filter "%s"', filter_name)
    if filter_name is not 'all':
        localizations = data['estimates'][data[filter_name]]
        locations = data['stimulus_locations'][data[filter_name]]
    else:
        localizations = data['estimates']
        locations = data['stimulus_locations']

    offsets = locations[:,1] - locations[:,0]
    
    signif_diff = np.abs(offsets) >= 0.01

    # Cut off where disparity is < .01 --- the relative distance doesn't make much sense there.
    locations = locations[signif_diff]
    localizations = localizations[signif_diff]
    offsets = offsets[signif_diff]

    # Compute the relative distance
    rel_localizations = (localizations - locations[:,0]) / offsets

    return nanmean(rel_localizations)

out_data = {
    'vis_data'   : compute_localization_statistics('is_visible'),
    'vis_mean'   : global_mean('is_visible'),
    'aud_data'   : compute_localization_statistics('is_noisy'),
    'aud_mean'   : global_mean('is_noisy'),
    'both_data'  : compute_localization_statistics('is_both'),
    'both_mean'   : global_mean('is_both'),
    'none_data'  : compute_localization_statistics('is_none'),
    'none_mean'   : global_mean('is_none'),
    'all_data'   : compute_localization_statistics('all'),
    'all_mean'   : global_mean('all'),
    'config'     : data['config'].item()
}

np.savez_compressed(file=args.outfile, **out_data)
