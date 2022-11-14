#!/usr/bin/env python
#
# Copyright (C) 2017 Joel S. Bader
# You may use, distribute, and modify this code
# under the terms of the Python Software Foundation License Version 2
# available at https://www.python.org/download/releases/2.7/license/
#
import argparse
import os
import fnmatch
import math
import string

import numpy as np
from scipy.stats import norm as norm_distribution
import matplotlib.cm as mpcm
import matplotlib.pyplot as plt

import logging
logging.basicConfig(format='%(levelname)s %(name)s.%(funcName)s: %(message)s')
logger = logging.getLogger('prior_vs_population')
logger.setLevel(logging.INFO)

def rsqbetween(log10tumor, log10organoid, zsq, wbratio):
    T = 10**log10tumor
    Nt = 10**log10organoid
    fac = zsq / (T - 1.0)
    val = (1.0 + (wbratio/Nt)) * fac / (1.0 + fac)
    return(val)

def rsqwithin(log10tumor, log10organoid, zsq, fracp):
    T = 10**log10tumor
    Nt = 10**log10organoid
    N = Nt * T
    fac = zsq / (fracp * (N - T - 1.0))
    val = fac / (1.0 + fac)
    return(val)
    
# zsq = N rsq/(1-rsq)
# zsq/N = rsq/(1-rsq)
# N/zsq = (1/rsq) - 1
# (N + zsq)/zsq = 1/rsq
# rsq = zsq/(N + zsq)
def get_lrsq(log10n, log10prior, pval, z2):
    n = 10**log10n
    prior = 10**log10prior
    newpval = pval * prior
    z1 = -norm_distribution.ppf(0.5 * newpval)
    zsq = (z1 - z2)**2
    rsq = zsq / (n + zsq)
    lrsq = np.log10(rsq)
    return(lrsq)
    
def write_powerplot(filename, lpop_lo, lpop_hi, lprior_lo, lprior_hi, pval, power):

    labelsize=15
    ticksize=12

    z1 = -norm_distribution.ppf(0.5 * pval)
    z2 = norm_distribution.ppf(1.0 - power)
    zsq = (z1-z2)**2
    logger.info('pval %g z1 %g power %g z2 %g zsq %g', pval, z1, power, z2, zsq)
    log10pop = np.linspace(lpop_lo, lpop_hi, 101)
    log10prior = np.linspace(lprior_lo, lprior_hi, 101)
    X, Y = np.meshgrid(log10pop, log10prior)
    Z = get_lrsq(X, Y, pval, z2)
    plt.figure(figsize=(10,8))
    plt.title(r'Critical $R^2$ for p-value %g, power %g' % (pval, power), fontsize=labelsize)
    plt.xlabel('Population size', fontsize=labelsize)
    plt.ylabel('Prior strength', fontsize=labelsize)
    xtickvals = [1.e2, 1.e3, 1.e4, 1.e5, 1.e6]
    xtickstrs = [ '{:,d}'.format(int(v)) for v in xtickvals ]
    xticklocs = np.log10(xtickvals)
    ytickvals = [1., 1.e1, 1.e2, 1.e3, 1.e4, 1.e5]
    ytickstrs = [ '{:,d}'.format(int(v)) for v in ytickvals ]
    yticklocs = np.log10(ytickvals)
    plt.xticks(xticklocs, xtickstrs, fontsize=ticksize)
    plt.yticks(yticklocs, ytickstrs, fontsize=ticksize)
    plt.contourf(X, Y, Z, 100, cmap=mpcm.jet)
    cbar = plt.colorbar()
    ctickvals = [ 5.e-5, 1.e-4, 2.e-4, 5.e-4, 1.e-3, 2.e-3, 5.e-3, 1.e-2, 1.e-2, 2.e-2, 5.e-2, 1.e-1 ]
    cticklocs = np.log10(ctickvals)
    ctickstrs = [ '{:f}'.format(v) for v in ctickvals ]
    cbar.set_ticks(cticklocs)
    cbar.set_ticklabels(ctickstrs)
    cbar.ax.set_xlabel(r'$R^2$')
    plt.contour(X, Y, Z, 100, cmap=mpcm.jet)
    plt.savefig(filename)
    plt.close()
    return None

# zsq = N rsq / (1-rsq)
# so rsq doesn't matter
def get_slope(s, pval, z2):
    sa = 1.0
    sb = s
    za = -norm_distribution.ppf(sa * pval)
    zb = -norm_distribution.ppf(sb * pval)
    zsqa = (za - z2)**2
    zsqb = (zb - z2)**2
    na = zsqa
    nb = zsqb
    m = np.log(sb/sa)/np.log(nb/na)
    f = na/nb
    return(m, f)

def write_calibrationplot(filename, pval, power):
    ls = np.linspace(0, 4, 1000)[1:] # exclude the last point
    S = 10**ls
    z2 = norm_distribution.ppf(1.0 - power)
    (M, Factor) = get_slope(S, pval, z2)
    # logger.info('S M %s', str(zip(S,M)))
    plt.figure(figsize=(10,8))

    s1 = 0
    s2 = 0
    s3 = 0
    for (mys, myf) in zip(S, Factor):
        if (s1 == 0) and (myf >= 1.2):
            s1 = mys
        if (s2 == 0) and (myf >= 1.4):
            s2 = mys
        if (s3 == 0) and (myf >= 2):
            s3 = mys
    logger.info('s1 %f s2 %f', s1, s2)
    logger.info('s3 %f', s3)

    # xtickvals = [ 1., 2., 5., 10., 20., 50., 100., 200., 500., 1000. ]
    # xticklocs = np.log10(xtickvals)
    # xtickstrs = [ '{:,d}'.format(int(v)) for v in xtickvals ]

    #labelsize=20
    #ticksize=18

    ticksize=24
    labelsize=28
    plt.plot(Factor, S, 'k-')
    plt.ylabel('Prior strength for equivalent power', fontsize=labelsize)
    plt.xlabel('Cohort fold-increase', fontsize=labelsize)
    plt.yscale('log')
    plt.xlim(1,2)
    plt.ylim(1,max(S))
    plt.plot([1.2, 1.2], [1,s1], 'k--')
    plt.plot([1.2, 1], [s1,s1], 'k--')
    plt.plot([1.4, 1.4], [1,s2], 'k--')
    plt.plot([1.4, 1.], [s2,s2], 'k--')
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    # plt.xticks(xticklocs, xtickstrs)

    plt.savefig(filename)

    plt.close()

    return None
    

def write_exponentplot(filename, pval, power):
    '''
    original version
    '''
    ls = np.linspace(0, 3, 1000)[1:] # exclude the last point
    S = 10**ls
    z2 = norm_distribution.ppf(1.0 - power)
    (M, Factor) = get_slope(S, pval, z2)
    # logger.info('S M %s', str(zip(S,M)))
    plt.figure(figsize=(12,5))

    xtickvals = [ 1., 2., 5., 10., 20., 50., 100., 200., 500., 1000. ]
    xticklocs = np.log10(xtickvals)
    xtickstrs = [ '{:,d}'.format(int(v)) for v in xtickvals ]

    plt.subplot(121)
    plt.plot(ls, Factor, 'k-')
    plt.xlabel('Prior strength')
    plt.ylabel('Population factor equivalent')
    plt.xticks(xticklocs, xtickstrs)

    plt.subplot(122)
    plt.plot(ls, -M, 'k-')
    plt.xticks(xticklocs, xtickstrs)
    plt.xlabel('Prior strength')
    plt.ylabel('Population size exponent')

    plt.savefig(filename)

    plt.close()

    return None

def write_exponentplot(filename, pval, power):

    labelsize=15
    ticksize=12
    plt.figure(figsize=(12,5))

    plt.subplot(121)
    
    ls = np.linspace(0, 3, 1000)[1:] # exclude the last point
    S = 10**ls
    z2 = norm_distribution.ppf(1.0 - power)
    (M, Factor) = get_slope(S, pval, z2)
    # logger.info('S M %s', str(zip(S,M)))
    
    xtickvals = [ 1., 2., 5., 10., 20., 50., 100., 200., 1000. ]
    xticklocs = np.log10(xtickvals)
    xtickstrs = [ '{:,d}'.format(int(v)) for v in xtickvals ]

    plt.plot(ls, -M, 'k-')
    plt.xticks(xticklocs, xtickstrs, fontsize=ticksize)
    plt.xlabel('Prior strength', fontsize=labelsize)
    plt.ylabel('Population size exponent', fontsize=labelsize)

    plt.subplot(122)

    ls = np.linspace(0, 4, 1000)[1:] # exclude the last point
    S = 10**ls
    z2 = norm_distribution.ppf(1.0 - power)
    (M, Factor) = get_slope(S, pval, z2)
    # logger.info('S M %s', str(zip(S,M)))

    s1 = 0
    s2 = 0
    for (mys, myf) in zip(S, Factor):
        if (s1 == 0) and (myf >= 1.2):
            s1 = mys
        if (s2 == 0) and (myf >= 1.4):
            s2 = mys
    logger.info('s1 %f s2 %f', s1, s2)

    
    plt.plot(S, Factor, 'k-')
    plt.xlabel('Prior strength', fontsize=labelsize)
    plt.ylabel('Cohort fold-increase', fontsize=labelsize)
    plt.xscale('log')
    plt.ylim(1,2)
    plt.xlim(1,max(S))
    plt.plot([1,s1], [1.2, 1.2],'k--')
    plt.plot([s1,s1], [1.2, 1],'k--')
    plt.plot([1,s2], [1.4, 1.4], 'k--')
    plt.plot([s2,s2], [1.4, 1.], 'k--')
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    # plt.xticks(xticklocs, xtickstrs)

    plt.savefig(filename)

    plt.close()

    return None

def main():
    
    parser = argparse.ArgumentParser(description='Invasive boundary image score, 2D',
                                     epilog='Sample call: see run.sh',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lpop_lo', help='log10(lower population size)', required=False, type=float, default=2)
    parser.add_argument('--lpop_hi', help='log10(higher population size)', required=False, type=float, default=6)
    parser.add_argument('--lprior_lo', help='log10(lower power strength)', required=False, type=float, default=0)
    parser.add_argument('--lprior_hi', help='log10(higher power strength)', required=False, type=float, default=4)
    parser.add_argument('--pval', help='p-value', required=False, type=float, default=5e-8)
    parser.add_argument('--power', help='power', required=False, type=float, default=0.8)
    parser.add_argument('--outdir', help='directory for writing output', required=False, default='./analytical')

    args = parser.parse_args()
    logger.info('args %s', str(args))

    # check that the output directory exists; if not, create it
    if (not os.path.isdir(args.outdir)):
        logger.info('creating output directory %s', args.outdir)
        os.makedirs(args.outdir)
    
    write_powerplot(os.path.join(args.outdir, 'prior_vs_population.pdf'), args.lpop_lo, args.lpop_hi, args.lprior_lo, args.lprior_hi, args.pval, args.power)
    write_calibrationplot(os.path.join(args.outdir, 'fig_calibration.pdf'), args.pval, args.power)
    write_exponentplot(os.path.join(args.outdir, 'prior_exponent.pdf'), args.pval, args.power)
    
if __name__ == "__main__":
    main()
