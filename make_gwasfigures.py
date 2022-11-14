#!/usr/bin/env python
#
# Copyright (C) 2017 Joel S. Bader
# You may use, distribute, and modify this code
# under the terms of the Python Software Foundation License Version 2
# available at https://www.python.org/download/releases/2.7/license/
#
import argparse
import os
import math

import numpy as np
import scipy as sp
from scipy.stats import norm

import matplotlib
matplotlib.use("TKAgg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as mpcm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as mlines
from matplotlib.patches import Polygon
import matplotlib.colors as mpcolors

import statsmodels.api as sm

import logging
logging.basicConfig(format='%(levelname)s %(name)s.%(funcName)s: %(message)s')
logger = logging.getLogger('make_designfigures')
logger.setLevel(logging.INFO)


# GWAS catalog columns (first column is 1, not 0)
DATE_PUBLISHED_COL = 4
PMID_COL = 2
SIZE_COL = 9
DISEASE_COL = 8
CHR_ID_COL = 12
CHR_POS_COL = 13
REPORTED_GENE_COL = 14
MAPPED_GENE_COL = 15
SNP_ID_CURRENT_COL = 24
PVALUE_COL = 28
PVALUE_MLOG_COL = 29
MAPPED_COL = 35

def isInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False    

def isFloat(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False    


MONTH_TO_DAYS = [ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 ]
def year_frac(m, d):
    # no zero-offset for m and d
    m1 = int(m) - 1 # to get the ending day of the previous month
    d1 = int(d)
    day = sum(MONTH_TO_DAYS[:m1]) + d1
    frac = float(day)/365.0
    if (frac > 1.0):
        frac = 1.0
    return(frac)

def date_to_year(datestr):
    # expect YYYY-MM-DD
    date_toks = datestr.split('-')
    assert(len(date_toks) == 3), 'bad date: %s' % datestr
    (y, m, d) = [ float(tok) for tok in date_toks ]
    frac = year_frac(m, d)
    year = y + frac
    assert((year >= 1990) and (year <= 2021)), 'bad year: %s %f' % (datestr, year)
    return(year)

NLINSPACE = 20
def fit_moore(year_list, value_list):
    log2value = [ np.log2(value) for value in value_list ]
    myfit = sm.OLS(log2value, sm.add_constant(year_list)).fit()
    year_min = np.min(year_list)
    year_max = np.max(year_list)
    year_fit = np.linspace(year_min - 0.5, year_max + 0.5, NLINSPACE)
    (b0, b1) = (myfit.params[0], myfit.params[1])
    value_fit = np.power(2.0, b0 + (b1 * year_fit))
    pvalue = myfit.pvalues[1]
    se = myfit.bse[1]
    tau = 1.0/b1
    tause = se * tau * tau
    return(tau, tause, pvalue, year_fit, value_fit)

def fit_moore_modelAnova(year_list, value_list):
    log2value = [ np.log2(value) for value in value_list ]

    ## constant model
    myfit_const = sm.OLS(log2value, np.ones(len(year_list))).fit()
    pvalue_const = myfit_const.pvalues[0]
    ssr_const = myfit_const.ssr
    df_const = myfit_const.df_resid

    ## linear model
    myfit_linear = sm.OLS(log2value, sm.add_constant(year_list)).fit()
    pvalue_linear = myfit_linear.pvalues[1]
    ssr_linear = myfit_linear.ssr
    df_linear = myfit_linear.df_resid

    ## quadratic model
    year2_list = [(x-np.min(year_list))**2 for x in year_list]
    year_list_quadratic = np.column_stack((year_list, year2_list))
    myfit_quadratic = sm.OLS(log2value, sm.add_constant(year_list_quadratic)).fit()
    pvalue_quadratic = myfit_quadratic.pvalues[2]
    ssr_quadratic = myfit_quadratic.ssr
    df_quadratic = myfit_quadratic.df_resid

    ## linear compare to constant
    fstat_linearConstant = (ssr_const-ssr_linear)/(df_const-df_linear)/(ssr_linear/df_linear)
    p_value_linearConstant = 1-sp.stats.f.cdf(fstat_linearConstant, df_const-df_linear, df_linear)

    ## quadratic compare to linear
    fstat_quadraticLinear = (ssr_linear-ssr_quadratic)/(df_linear-df_quadratic)/(ssr_quadratic/df_quadratic)
    p_value_quadraticLinear = 1-sp.stats.f.cdf(fstat_quadraticLinear, df_linear-df_quadratic, df_quadratic) 

    return(pvalue_const, ssr_const, df_const, \
        pvalue_linear, ssr_linear, df_linear, \
        pvalue_quadratic, ssr_quadratic, df_quadratic, \
        fstat_linearConstant, p_value_linearConstant, \
        fstat_quadraticLinear, p_value_quadraticLinear)   

def read_trait_abbrev(filename):
    fp = open(filename, 'r')
    header = fp.readline()
    ret = dict()
    for line in fp:
        toks = line.strip().split('\t')
        (trait, abbrev, dx, dy) = toks[:4]
        (dx, dy) = (float(dx), float(dy))
        assert(trait not in ret), 'repeated trait: %s' % trait
        ret[trait] = (abbrev, dx, dy)
    return(ret)

def plot_trait_moore(plotfile, trait, year_list, size_list, nsignif_list, year_fit, size_fit, nsignif_fit,
                     size_tau, size_se, size_pval,
                     nsignif_tau, nsignif_se, nsignif_pval):
    logger.info('plotting %s', trait)
    plt.figure()
    fig, ax1 = plt.subplots(1,1,figsize=(11,8))
    markersize = 80
    ax1.scatter(year_list, size_list, marker='o', facecolors='tan', edgecolors='k', s=markersize, linewidth=2, label='Cohort size')
    ax1.scatter(year_list, nsignif_list, marker='^', facecolors='k', edgecolors='k', s=markersize, label='Loci')
    ax1.plot(year_fit, size_fit, 'k--', label=r'Cohort doubling time %.1f $\pm$ %.1f yrs, p = %.3g' % (size_tau, size_se, size_pval))
    ax1.plot(year_fit, nsignif_fit, 'k-', label=r'Loci doubling time %.1f $\pm$ %.1f yrs, p = %.3g' % (nsignif_tau, nsignif_se, nsignif_pval))
    legendsize=20
    ticksize=24
    labelsize=28
    ax1.set_title(trait, fontsize=labelsize)
    ax1.set_xlabel('Year', fontsize=labelsize)
    ax1.set_ylabel('Count', fontsize=labelsize)
    ax1.set_yscale('log')
    ax1.set_ylim(1,1.e6)
    ax1.tick_params(labelsize=ticksize)
    ax1.legend(loc='upper left', frameon=False, fontsize=legendsize)
    fig.tight_layout()
    plt.savefig(plotfile)
    plt.close('all')
    return None

def plot_doublingtime(moore_file, plot_file, trait_abbrev):
    logger.info('reading data from %s and plotting to %s', moore_file, plot_file)
    (TMIN, TMAX) = (0., 2.75)
    textsize = 20
    ticksize = 24
    labelsize = 24
    plt.figure()
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax.axis('equal')
    ax.set(xlim=(TMIN,TMAX), ylim=(TMIN,TMAX))
    myticks = [0., 0.5,1.0,1.5,2.0,2.5]
    ax.set_xticks(myticks)
    ax.set_yticks(myticks)
    fp = open(moore_file, 'r')
    header = fp.readline()
    MAX_SE = 0.5
    sizevec = [ ]
    locivec = [ ]
    dxvec = [ ]
    dyvec = [ ]
    abbrevvec = [ ]
    colorvec = [ ]
    pvec = [ ]

    for line in fp:
        toks = line.strip().split('\t')
        assert(len(toks) == 16), 'bad token count %d: %s' % (len(toks), line)
        (trait, diffstr) = tuple([ toks[i-1] for i in [ 1, 16 ] ])
        (ntot, ninform, nsignif) = tuple([ int(toks[i-1]) for i in [ 2, 3, 7 ] ])
        (sizetime, sizese, locitime, locise) = tuple([ float(toks[i-1]) for i in [ 8, 9, 11, 12 ] ])
        (abbrev, dx, dy) = trait_abbrev.get(trait, (trait, 0.0, 0.0))
        if (sizese > MAX_SE) or (locise > MAX_SE) or (nsignif < 10):
            logger.info('... skipping %s %s', abbrev, trait)
            continue
        logger.info('%s %s %d %d %f %f %s', trait, abbrev, ntot, ninform, sizetime, locitime, diffstr)
        if (sizetime > TMAX) or (locitime > TMAX):
            logger.info('... WARNING %s is outside bounds', abbrev)
        mycolor = 'black'
        if (diffstr == 'discovering'):
            mycolor = 'green'
        elif (diffstr == 'saturating'):
            mycolor = 'red'

        # keep for plotting
        sizevec.append(sizetime)
        locivec.append(locitime)
        dxvec.append(dx)
        dyvec.append(dy)
        abbrevvec.append(abbrev)
        colorvec.append(mycolor)
        pvec.append(float(toks[15-1]))
        # ax.text(sizetime + dx, locitime + dy, abbrev, color=mycolor, ha='center', va='center', fontsize=textsize)
        
    fp.close()
    ax.set_xlabel('Years for cohort to double', fontsize=labelsize)
    ax.set_ylabel('Years for loci to double', fontsize=labelsize)
    ax.tick_params(labelsize=ticksize)

    logger.info('size, locus %s', str(sorted(list(zip(sizevec, locivec)))))
    logger.info('total amount of plottig: %s', str(len(sizevec)))
    logger.info('count discovering: %s', str(colorvec.count('green')))
    logger.info('count saturating: %s', str(colorvec.count('red')))
    logger.info('count equal: %s', str(colorvec.count('black')))
    pCorrectvec = [ x<(0.05/len(sizevec)) for x in pvec ]
    logger.info('count significant corrected: %s', str(sum(pCorrectvec)))
    logger.info('significant corrected trait: %s', ','.join([abbrevvec[i] for i in range(len(pCorrectvec)) if pCorrectvec[i]]))
    (size_mean, size_std) = (np.mean(sizevec), np.std(sizevec, ddof=1))
    (loci_mean, loci_std) = (np.mean(locivec), np.std(locivec, ddof=1))
    logger.info('size %f +/- %f', size_mean, size_std)
    logger.info('loci %f +/- %f', loci_mean, loci_std)

    x_fit = np.linspace(TMIN+0.5, TMAX-0.25, 20)

    # proportional fit
    propfit = sm.OLS(locivec, sizevec).fit()
    logger.info('*** proportional fit, zero intercept***\n%s\n*** end fit ***', str(propfit.summary()))
    mystr = r'loci = (%.2f $\pm$ %.2f) $\times$ cohort, $R^2$ = %.2f, $p$-value = %.2g' % (propfit.params[0], propfit.bse[0], propfit.rsquared_adj, propfit.pvalues[0])
    
    # linear fit
    linfit = sm.OLS(locivec,sm.add_constant(sizevec)).fit()
    logger.info('*** fit ***\n%s\n*** end fit ***', str(linfit.summary()))
    y_fit = linfit.params[0] + (linfit.params[1] * x_fit)
    
    # text goes last to be on top of fitting line
    for (x, dx, y, dy, abbrev, mycolor) in zip(sizevec, dxvec, locivec, dyvec, abbrevvec, colorvec):
        ax.text(x + dx, y + dy, abbrev, color=mycolor, ha='center', va='center', fontsize=textsize)

    legendsize = 14
    ax.legend(loc='upper left', frameon=False, fontsize=legendsize)
    
    plt.savefig(plot_file)
    plt.close('all')

    a1 = sm.stats.anova_lm(propfit, linfit)
    a2 = sm.stats.anova_lm(linfit, propfit)

    logger.info('*** anova prop lin ***\n%s\n', str(a1))
    logger.info('*** anova lin prop ***\n%s\n', str(a2))
    
    return None
                      
def get_trait_moore(moore_file, plot_dir, trait_pmid_summary, compare_dir, MIN_INFORMATIVE=3, SIGNIF=0.05):
    trait_list = sorted(trait_pmid_summary.keys())
    fp = open(moore_file, 'w')
    fp.write('\t'.join(['Trait', 'n_tot', 'n_informative',
                        'recent_pmid', 'recent_year', 'recent_size', 'recent_nsignif',
                        'size_tau', 'size_se', 'size_pval', 'nsignif_tau', 'nsignif_se', 'nsignif_pval', 'taudiff_z', 'taudiff_p', 'taudiff_str']) + '\n')

    modelCompareSizeFile = os.path.join(compare_dir, 'modelCompare_size.txt')
    fp_modelCompare_size = open(modelCompareSizeFile, 'w')
    fp_modelCompare_size.write('\t'.join(['Trait', \
        'pvalue_const', 'ssr_const', 'df_const', \
        'pvalue_linear', 'ssr_linear', 'df_linear', \
        'pvalue_quadratic', 'ssr_quadratic', 'df_quadratic', \
        'fstat_linearConstant', 'p_value_linearConstant', \
        'fstat_quadraticLinear', 'p_value_quadraticLinear'])+'\n')

    modelCompareNsignifFile = os.path.join(compare_dir, 'modelCompare_nsignif.txt')
    fp_modelCompare_nsignif = open(modelCompareNsignifFile, 'w')
    fp_modelCompare_nsignif.write('\t'.join(['Trait',\
        'pvalue_const', 'ssr_const', 'df_const', \
        'pvalue_linear', 'ssr_linear', 'df_linear', \
        'pvalue_quadratic', 'ssr_quadratic', 'df_quadratic', \
        'fstat_linearConstant', 'p_value_linearConstant', \
        'fstat_quadraticLinear', 'p_value_quadraticLinear'])+'\n')

    for trait in trait_list:
        # keep the studies that increase the number of findings
        # use date as the order
        pmid_sorted = sorted(trait_pmid_summary[trait].keys())
        n_tot = len(pmid_sorted)
        if (n_tot < MIN_INFORMATIVE):
            continue
        logger.info('%d total studies for %s', n_tot, trait)
        recs = [ (trait_pmid_summary[trait][pmid]['date_published'], pmid) for pmid in pmid_sorted ]
        recs = sorted(recs)
        year_list = [ ]
        size_list = [ ]
        nsignif_list = [ ]
        pmid_list = [ ]
        best_cnt = 0
        for (date, pmid) in recs:
            (size, nsignif) = tuple([ trait_pmid_summary[trait][pmid][f] for f in ['size_effective', 'nsignif'] ])
            if nsignif < best_cnt:
                continue
            best_cnt = nsignif
            year = date_to_year(date)
            year_list.append(year)
            size_list.append(size)
            nsignif_list.append(nsignif)
            pmid_list.append(pmid)
        n_informative = len(year_list)
        # logger.info('* %d studies, %d loci for %s', n_informative, best_cnt, trait)
        if (n_informative < MIN_INFORMATIVE):
            continue
        if (best_cnt < MIN_INFORMATIVE):
            continue
        (size_tau, size_se, size_pval, year_fit, size_fit) = fit_moore(year_list, size_list)
        (nsignif_tau, nsignif_se, nsignif_pval, year1_fit, nsignif_fit) = fit_moore(year_list, nsignif_list)

        diff_value = size_tau - nsignif_tau
        diff_se = math.sqrt( (size_se**2) + (nsignif_se**2) )
        diff_z = diff_value / diff_se
        diff_p = 2.0 * norm.sf(abs(diff_z))
        diff_str = 'insufficient_data'
        if (size_pval <= SIGNIF) and (nsignif_pval <= SIGNIF):
            diff_str = 'tie'
            if (diff_p <= SIGNIF): 
                if (diff_z < 0.0): # size is doubling faster, so saturating
                    diff_str = 'saturating'
                elif (diff_z > 0.0): # findings are doubling faster
                    diff_str = 'discovering'
        if (diff_str == 'insufficient_data'):
            continue

        ## model comparison
        modelCompare_size = fit_moore_modelAnova(year_list, size_list)
        fp_modelCompare_size.write('\t'.join([str(trait)] + [str(x) for x in modelCompare_size])+'\n')
        modelCompare_nsignif = fit_moore_modelAnova(year_list, nsignif_list)
        fp_modelCompare_nsignif.write('\t'.join([str(trait)] + [str(x) for x in modelCompare_nsignif])+'\n')

        fp.write('\t'.join([ str(f) for f in [ trait, n_tot, n_informative,
                                               pmid_list[-1], year_list[-1], size_list[-1], nsignif_list[-1],
                                               size_tau, size_se, size_pval, nsignif_tau, nsignif_se, nsignif_pval,
                                               diff_z, diff_p, diff_str ] ]) + '\n')
        plotfile = os.path.join(plot_dir, trait + '.pdf')
        plot_trait_moore(plotfile, trait, year_list, size_list, nsignif_list, year_fit, size_fit, nsignif_fit,
                     size_tau, size_se, size_pval,
                     nsignif_tau, nsignif_se, nsignif_pval)
    fp.close()
    return None
    

def read_chr(chrfile):
    fp = open(chrfile, 'r')
    header = fp.readline()
    chr_list = list()
    chr_length = dict()
    chr_begin = dict()
    chr_end = dict()
    for line in fp:
        toks = line.strip().split()
        assert(len(toks) == 4), 'bad line: ' + line
        (chrname, chrlen) = toks[:2]
        assert(chrname not in chr_list), 'repeated chromosome: ' + chrname
        chr_list.append(chrname)
        chr_length[chrname] = int(chrlen)
    fp.close()
    c = chr_list[0]
    chr_begin[ c ] = 1
    chr_end[ c ] = chr_length[ c ]
    for i in range(1,len(chr_list)):
        c = chr_list[i]
        b = chr_list[i-1]
        # logger.info('chr %s <- %s', c, b)
        chr_begin[c] = chr_end[b] + 1
        chr_end[c] = chr_begin[c] + chr_length[c] - 1
    #logger.info('chr info')
    #for c in chr_list:
    #    logger.info('%s %d %d %d %d', c, chr_begin[c], chr_end[c], chr_length[c], chr_end[c] - chr_begin[c] + 1 - chr_length[c])
    return(chr_list, chr_begin, chr_end, chr_length)


def plot_manhattan_threepanel(plotfile, trait, trait_rows, chr_list, chr_begin, chr_end, chr_length):

    signif = -math.log10(5.e-8)
    years = [ 2010, 2015, 2020 ]
    nyears = len(years)
    color_list = [ 'darkblue' , 'darkgreen' ]
    color_list_low = [ 'cornflowerblue', 'lightgreen' ] 
    ncolor = len(color_list)

    chr_to_int = dict()
    cnt = 0
    for c in chr_list:
        chr_to_int[c] = cnt
        cnt = cnt + 1

    # create a tuple with date published, chromosome, xvalue, yvalue, mapped gene
    data_list = [ ]
    for line in trait_rows:
        toks = line.strip().split('\t')
        mytuple = tuple([ toks[i-1] for i in [DATE_PUBLISHED_COL, SNP_ID_CURRENT_COL, CHR_ID_COL, CHR_POS_COL, PVALUE_MLOG_COL, MAPPED_GENE_COL]])
        (mydate, mysnp, mychr, mypos, mymlog, mygene) = mytuple
        if mychr not in chr_list:
            # logger.info('bad chr: %s', c)
            continue
        if not isInt(mypos):
            logger.info('bad pos: %s', mypos)
            continue
        myx = int(mypos) + chr_begin[mychr]
        if not isFloat(mymlog):
            logger.info('bad mlog: %s', mymlog)
            continue
        myy = float(mymlog)
        gene_toks = mygene.split(' ')
        firstgene = gene_toks[0].strip(',;')
        data_list.append( (mydate, mysnp, mychr, myx, myy, firstgene) )
    data_sorted = sorted(data_list)

    ## count the numbers of significant hits across years
    year_cnt = dict()
    for year_cutoff in years:
        year_cnt[year_cutoff] = []

    for rec in data_sorted:
        (mydate, mysnp, mychr, myx, mymlog, mygene) = rec
        if mysnp == '':
            continue
        if mymlog <= signif:
            continue
        (yyyy, mm, dd) = mydate.split('-')
        for year_cutoff in years:
            if (int(yyyy) > year_cutoff):
                continue
            if mysnp in year_cnt[year_cutoff]:
                continue
            year_cnt[year_cutoff].append(mysnp)
    year_cnt_num = dict()
    for year_cutoff in years:
        year_cnt_num[year_cutoff] = len(year_cnt[year_cutoff])

    logger.info('Trait: %s traits has numbers of significants \n %s', trait, str(year_cnt_num))

    last_chr = chr_list[-1]
    XMIN = 0
    XMAX = chr_end[ last_chr ]
    
    MLOGMAX = 15
    YMIN = 4
    YMAX = MLOGMAX + 5

    tickfontsize = 10
    bigsize = 16
    mediumsize = 14
    xtickposn = [ chr_begin[c] + (0.5 * chr_length[c]) for c in chr_list ]
    ytickposn = [ 4, 6, 8, 10, 12, 14 ]
    yticklabel = [ str(y) for y in ytickposn ]
    
    fig, axs = plt.subplots(nrows=nyears, sharex=True, figsize=(15,15))
    fig.suptitle('%s GWAS loci' % trait, fontsize=bigsize)
    for i in range(nyears):
        year_cutoff = years[i]
        axs[i].set_xlim(XMIN, XMAX)
        axs[i].set_ylim(YMIN, YMAX)
        axs[i].set_title('Through %d' % years[i], fontsize=mediumsize)
        axs[i].axhline(signif, xmin=XMIN, xmax=XMAX, color='gray', linestyle='dashed')
        axs[i].set_xticks(xtickposn)
        axs[i].set_ylabel(r'$-\log_{10}$' + ' p-value', fontsize=mediumsize)
        axs[i].set_yticks(ytickposn)
        axs[i].set_yticklabels(yticklabel, fontsize=tickfontsize)
        if (i == (nyears-1)):
            axs[i].set_xlabel('Chromosome', fontsize=mediumsize)
            axs[i].set_xticklabels(chr_list, fontsize=tickfontsize)
        snp_to_mlog = dict()
        snp_to_x = dict()
        snp_to_y = dict()
        snp_to_color = dict()
        gene_to_x = dict()
        gene_to_mlog = dict()
        for rec in data_sorted:
            (mydate, mysnp, mychr, myx, mymlog, mygene) = rec
            if mysnp == '':
                continue
            (yyyy, mm, dd) = mydate.split('-')
            if (int(yyyy) > year_cutoff):
                break
            if mymlog > snp_to_mlog.get(mysnp, -1):
                snp_to_mlog[mysnp] = mymlog
                snp_to_x[mysnp] = myx
                icolor = chr_to_int[mychr] % ncolor
                snp_to_color[mysnp] = color_list[ icolor ] if mymlog >= signif else color_list_low[ icolor ]
            if (mymlog >= MLOGMAX) and (mygene is not '') and (mymlog > gene_to_mlog.get(mygene, -1)):
                gene_to_mlog[mygene] = mymlog
                gene_to_x[mygene] = myx
        snps = sorted(snp_to_mlog.keys())
        xvec = [ snp_to_x[s] for s in snps ]
        yvec = [ min(snp_to_mlog[s], MLOGMAX) for s in snps ]
        cvec = [ snp_to_color[s] for s in snps ]
        axs[i].scatter(xvec, yvec, color=cvec)    
        genes = sorted(gene_to_x.keys())
        xvec = [ gene_to_x[g] for g in genes ]
        yvec = [ MLOGMAX + 0.5 ] * len(genes)
           
    plt.savefig(plotfile, bbox_inches='tight')
    plt.close()

    return None
    
    for (y, color) in zip(years_reversed, colors_reversed):
        
        xvec = [ ]
        yvec = [ ]
        cvec = [ ]
        text_x = [ ]
        text_y = [ ]
        text_str = [ ]
        for line in year_to_rows[y]:
            toks = line.strip().split('\t')
            (c, p, mlog, gene) = tuple([ toks[i-1] for i in [CHR_ID_COL, CHR_POS_COL, PVALUE_MLOG_COL, MAPPED_GENE_COL]])
            if c not in chr_list:
                # logger.info('bad chr: %s', c)
                continue
            if not isInt(p):
                logger.info('bad pos: %s', p)
                continue
            myx = int(p) + chr_begin[c]
            if not isFloat(mlog):
                logger.info('bad mlog: %s', mlog)
                continue
            myy = float(mlog)
            if myy >= YMAX:
                myy = YMAX
                if ' ' in gene:
                    continue
                text_x.append(myx)
                text_y.append(myy + 0.5)
                text_str.append(gene)    

            xvec.append(myx)
            yvec.append(myy)
            cvec.append(color)

        plt.scatter(xvec, yvec, color=cvec, label=str(y))
        if (y == 2020):
            for (x, y, s) in zip(text_x, text_y, text_str):
                plt.text(x, y, s, ha='center', va='bottom', rotation=90)
    return None

    
def plot_manhattan(plot_dir, subset, trait_abbrev, trait_rows, chr_list, chr_begin, chr_end, chr_length):
    for trait in sorted(subset):
        abbrev = trait_abbrev.get(trait, [trait,])[0]
        for mychr in ['/', '\\']:
            abbrev = abbrev.replace(mychr,'_')
        threepanel = os.path.join(plot_dir, abbrev + '.pdf')
        nrows = len(trait_rows[trait])
        logger.info('manhattan plot: %d records for %s', nrows, threepanel)
        if (nrows < 10):
            logger.info('... skipping')
        else:
            plot_manhattan_threepanel(threepanel, trait, trait_rows[trait], chr_list, chr_begin, chr_end, chr_length)
        
    return None

def parse_size(sizestr):
    size_cases = 0
    size_controls = 0
    size_other = 0
    size_total = 0
    groups = sizestr.split(', ')
    for g in groups:
        toks = g.split()
        category = toks[-1]
        if category not in ['cases', 'controls']:
            category = 'other'
        for t in toks:
            nocomma = t.replace(',','')
            if isInt(nocomma):
                size = int(nocomma)
                size_total += size
                if (category == 'cases'):
                    size_cases += size
                elif (category == 'controls'):
                    size_controls += size
                else:
                    size_other += size
    size_effective = size_total
    if (size_cases * size_controls > 0):
        harm_mean = 2.0/( (1.0/float(size_cases)) + (1.0/float(size_controls)) )
        size_effective = 2.0 * harm_mean
    # logger.info('%f %f %f %f %f <- %s', size_cases, size_controls, size_other, size_total, size_effective, sizestr)
    assert(size_total == size_cases + size_controls + size_other), 'bad count'
    return(size_cases, size_controls, size_other, size_total, size_effective)

def write_trait_pmid_summary(outfile, trait_pmid_summary):
    mykeys = [ 'trait', 'pmid' ]
    myfields = ['date_published', 'size_cases', 'size_controls', 'size_other', 'size_total', 'size_effective', 'nsignif']
    traits = sorted(trait_pmid_summary.keys())
    logger.info('writing trait_pmid_summary to %s', outfile)
    fp = open(outfile, 'w')
    fp.write('\t'.join(mykeys + myfields) + '\n')
    for t in traits:
        pmids = sorted(trait_pmid_summary[t].keys())
        for p in pmids:
            toks = [ t, p ] + [ str(trait_pmid_summary[t][p][f]) for f in myfields ]
            fp.write('\t'.join(toks) + '\n')
    fp.close()
    return None

def write_trait_rows(outfile, trait_rows):
    fp = open(outfile, 'w')
    traits = sorted(trait_rows.keys())
    for t in traits:
        rows = sorted(trait_rows[t])
        fp.write('*** %d rows for %s ***\n' % (len(rows), t))
        for r in rows:
            fp.write(r)
    fp.close()
    return None

def get_trait_pmid_summary(catalogfile, traittype):
    trait_col = DISEASE_COL if traittype == 'disease' else MAPPED_COL
    signif = -math.log10(5.e-8)
    logger.info('reading from %s', catalogfile)
    fp = open(catalogfile, 'r')
    header = fp.readline()
    toks = header.strip().split('\t')
    nexpect = len(toks)
    trait_pmid_summary = dict()
    trait_rows = dict()
    for line in fp:
        toks = line.strip().split('\t')
        assert(len(toks) == nexpect), 'error, bad token count, expected %d: %s' & (nexpect, line)
        (date_published, pmid, sizestr, trait, mlogp) = [ toks[i-1] for i in [ DATE_PUBLISHED_COL, PMID_COL, SIZE_COL, trait_col, PVALUE_MLOG_COL ] ]
        if trait not in trait_rows:
            trait_rows[trait] = [ ]
        trait_rows[trait].append(line)
        if float(mlogp) < signif:
                continue
        if trait not in trait_pmid_summary:
                trait_pmid_summary[trait] = dict()
        if pmid not in trait_pmid_summary[trait]:
                trait_pmid_summary[trait][pmid] = dict()
                # get size_case, size_control, size_other, size_total
                # logger.info('getting size for trait %s pmid %s sizestr %s', trait, pmid, sizestr)
                (size_cases, size_controls, size_other, size_total, size_effective) = parse_size(sizestr)
                trait_pmid_summary[trait][pmid]['size_cases'] = size_cases
                trait_pmid_summary[trait][pmid]['size_controls'] = size_controls
                trait_pmid_summary[trait][pmid]['size_other'] = size_other
                trait_pmid_summary[trait][pmid]['size_total'] = size_total
                trait_pmid_summary[trait][pmid]['size_effective'] = size_effective
                trait_pmid_summary[trait][pmid]['date_published'] = date_published
                trait_pmid_summary[trait][pmid]['nsignif'] = 0
        trait_pmid_summary[trait][pmid]['nsignif'] += 1
    fp.close()
    ntrait = len(trait_pmid_summary)
    nstudy = 0
    nsignif = 0
    for t in trait_pmid_summary:
        nstudy += len(trait_pmid_summary[t].keys())
        for p in trait_pmid_summary[t]:
            nsignif += trait_pmid_summary[t][p]['nsignif']
    logger.info('%d traits, %d studies, %d signif', ntrait, nstudy, nsignif)
    return(trait_pmid_summary, trait_rows)
    
def main():
    
    parser = argparse.ArgumentParser(description='Information content and functional units',
                                     epilog='No epilog',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--catalog', help='gwas catalog', required=False, default='data_catalog/gwas_catalog_v1.0.2-associations_e100_r2020-06-13.tsv')
    parser.add_argument('--traittype', help='disease or mapped', choices=['disease', 'mapped'], required=False, default='disease')
    parser.add_argument('--traitabbrev_file', help='trait abbreviations', required=False, default='data_catalog/trait_abbreviation.txt')
    parser.add_argument('--chrfile', help='chromosome information', required=False, default='data_catalog/chr_length_grch38p13.txt')
    parser.add_argument('--logfile', help='log file with linear models', required=False, default='logfile.txt')
    parser.add_argument('--outdir', help='directory for writing output', required=False, default='./results')
    args = parser.parse_args()
    
    
    # check that the output directory exists; if not, create it
    if (not os.path.isdir(args.outdir)):
        logger.info('creating output directory %s', args.outdir)
        os.makedirs(args.outdir)
    
    # create a study summary
    # each pmid can have multiple traits
    # traits are represented in 3 columns: disease trait, mapped trait, and EFO
    # disease trait seems the most specific, mapped trait is more general, EFO can have multiple
    # let user choose whether to analyze based on disease or mapped
    # Fields to keep are date added, cohort size, number of significant associations
    (trait_pmid_summary, trait_rows) = get_trait_pmid_summary(args.catalog, args.traittype)
    write_trait_pmid_summary(os.path.join(args.outdir, args.traittype + '_pmid_summary.txt'), trait_pmid_summary)
    # write_trait_rows(os.path.join(args.outdir, args.traittype + '_rows.txt'), trait_rows)

    # chromosome information for manhattan plot
    (chr_list, chr_begin, chr_end, chr_length) = read_chr(args.chrfile)
    trait_abbrev = read_trait_abbrev(args.traitabbrev_file)
    subset = set(trait_abbrev.keys())

    # create moores law plots for each trait
    moore_file = os.path.join(args.outdir, 'disease_mooreslaw.txt')
    doubling_plot = os.path.join(args.outdir, 'fig_doublingtime.pdf')
    plot_dir = os.path.join(args.outdir, 'plot_moore')
    compare_dir = os.path.join(args.outdir, 'model_compare')
    if (not os.path.isdir(plot_dir)):
        logger.info('creating plot directory %s', plot_dir)
        os.makedirs(plot_dir)
    if (not os.path.isdir(compare_dir)):
        logger.info('creating compare directory %s', compare_dir)
        os.makedirs(compare_dir)
    get_trait_moore(moore_file, plot_dir, trait_pmid_summary, compare_dir)
    plot_doublingtime(moore_file, doubling_plot, trait_abbrev)


    plot_dir = os.path.join(args.outdir, 'plot_manhattan')
    if (not os.path.isdir(plot_dir)):
        os.makedirs(plot_dir)
    plot_manhattan(plot_dir, subset, trait_abbrev, trait_rows, chr_list, chr_begin, chr_end, chr_length)
    return
     

if __name__ == "__main__":
    main()
