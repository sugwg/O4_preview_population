from pycbc.mchirp_area import calc_probabilities
from scipy.optimize import minimize, basinhopping, dual_annealing
from pycbc.conversions import mass1_from_mchirp_q, mchirp_from_mass1_mass2
from pycbc.cosmology import redshift
from astropy.utils.data import download_file
import logging
import json


def mass1_from_mchirp_mass2(mchirp, mass2):
    b = mass2
    c = mchirp

    d = (27 * b ** 5 - 4 * c ** 5)
    e = (b ** 9 * c ** 10 * d) ** 0.5
    f = (9 * b ** 7 * c ** 5 + (3 ** 0.5) * e)

    return (
    (2 ** (2 / 3) * 3 ** (1 / 3) * b ** 3 * c ** 5 +
     (f ** (1 / 3)) ** 2) /
    (2 ** (1 / 3) * 3 ** (2 / 3) * b ** 3 *
     f ** (1 / 3))
    )

def mchirp_from_source_class(superevent):
    # Parameters used for the source classification,
    # This may need to be periodically confirmed
    classes = ['BNS', 'BBH', 'NSBH']
    args = {'mass_limits': {'max_m1':45.0, 'min_m2':1.0},
            'mass_bdary':{'ns_max': 3.0, 'gap_max': 3.0},
            'estimation_coeff':{'a0': 0.74899, 'b0':-0.51557,
                                'b1':-0.322195, 'm0':.01},
            'lal_cosmology':True,
            'mass_gap':False,
            'mass_gap_separate':False}

    sid = superevent['superevent_id']

    root_url = superevent['links']['files']
    fdir = download_file(root_url, cache=True)
    flist = json.load(open(fdir, 'r'))
    pastro_file = None
    for pfile in ['pycbc.p_astro.json', 'spiir.p_astro.json']:
        if pfile in flist:
            pastro_file = download_file(flist[pfile], cache=True)
            break
    else:
        logging.info(f"Can't source classification for {sid}")
        return {'mchirp': None, 'tol': None, 'pfile': pfile}

    target = json.load(open(pastro_file, 'r'))
    ptotal = sum([target[k] for k in classes])
    target = {k:target[k] / ptotal for k in classes}

    if target['BBH'] >= .999:
        logging.info("Source classification uninformative")
        return {'mchirp': None, 'tol': None, 'pfile': pfile}

    if 'gstlal' in pfile:
        print(target)

    def findme(x):
        mc = x
        snr = 10
        dist = 0
        v = calc_probabilities(mc, snr, dist, args)
        met = 0
        for k in target:
            met += abs((v[k] - target[k]))**2.0
        return met

    tol = 1
    i = 1
    while tol >= 1e-7:
        r = dual_annealing(findme, bounds=[(1, 8.837)], maxfun=1000, maxiter=5 * i)
        tol = r['fun']
        i += 1
    mc = r['x'][0]
    logging.info(f"{mc}: {tol}")
    return {'mchirp': mc, 'tol': tol, 'pfile': pfile}

