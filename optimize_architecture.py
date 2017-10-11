import numpy as np
import numpy.random as npr
import astropy.stats as aps
import astropy.io.fits as apf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as mp
import subprocess as sp
import scipy.stats as ss
import scipy.signal as si
import matplotlib.cm as mpcm
import matplotlib.colors as mpc
import sklearn.neural_network as sk
import random as ra

######################################################################

def allocate_jobs(n_jobs, n_procs=1, rank=0):
    n_j_allocated = 0
    for i in range(n_procs):
        n_j_remain = n_jobs - n_j_allocated
        n_p_remain = n_procs - i
        n_j_to_allocate = n_j_remain / n_p_remain
        if rank == i:
            return range(n_j_allocated, \
                         n_j_allocated + n_j_to_allocate)
        n_j_allocated += n_j_to_allocate

def allocate_jobs_inc_time(n_jobs, n_procs=1, rank=0):
    allocated = []
    for i in range(n_jobs):
        if rank == np.mod(n_jobs-i, n_procs):
            allocated.append(i)
    return allocated

def complete_array(target_distrib, use_mpi=False):
    if use_mpi:
        target = np.zeros(target_distrib.shape)
        mpi.COMM_WORLD.Reduce(target_distrib, target, op=mpi.SUM, \
                              root=0)
    else:
        target = target_distrib
    return target

######################################################################

# plotting settings
lw = 1.5
mp.rc('font', family = 'serif')
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw
cm = mpcm.get_cmap('plasma')

# useful constants
d2s = 24.0 * 3600.0

# settings
dataset = 'gloess' # 'gloess' or 'crts' or 'sim'
use_mpi = True
include_period = True
split_to_train = True
std_by_bin = False
test_training_length = False
n_rpt = 20
if dataset == 'gloess':
    set_size = 4
elif dataset == 'crts':
    set_size = 19 # 9
elif dataset == 'sim':
    set_size = 250 # 9
base = dataset
if include_period:
    base += '_inc_per'
if use_mpi:
    import mpi4py.MPI as mpi
    n_procs = mpi.COMM_WORLD.Get_size()
    rank = mpi.COMM_WORLD.Get_rank()
else:
    n_procs = 1
    rank = 0

# switch on dataset
if dataset == 'gloess':

    # dataset settings
    # @TODO: might be able to push n_bins higher for this cadence
    data_dir = 'data/gloess/'
    n_bins = 50

    # get training stars
    cat_hdulist = apf.open(data_dir + 'gloess_cat.fit')
    cols = cat_hdulist[1].columns
    data = cat_hdulist[1].data
    ids = data['Name']
    fehs = data['__Fe_H_']
    taus = data['FPer'] * d2s
    types = data['RRL']

    # check for correct type
    rrab = (types == 'RRab')
    ids = ids[rrab]
    fehs = fehs[rrab]
    taus = taus[rrab]
    n_lc = len(ids)

    # period distribution
    tau_mean = np.mean(taus)
    tau_std = np.std(taus)

    # plot colours set by metallicities
    feh_min = np.min(fehs)
    feh_max = np.max(fehs)
    feh_cols = (fehs - feh_min) / (feh_max - feh_min)

    # read in lightcurves
    cat_hdulist = apf.open(data_dir + 'gloess_lcs.fit')
    cols = cat_hdulist[1].columns
    data = cat_hdulist[1].data
    binned_med_lcs = []
    binned_mean_lcs = []
    binned_mean_lc_stds = []
    if rank == 0:
        fig_sum, axes_sum = mp.subplots(1, 2, figsize=(16,5))
    for i in range(n_lc):

        # extract quantities of interest
        inds = (data['Name'] == ids[i]) & (data['Flt'] == 'V')
        phase = data['Phase'][inds]
        mag = data['mag'][inds]

        # calculate some binned statistics; no mag errors available
        bins = np.linspace(0, 1, n_bins + 1)
        meds, edges, i_bins = ss.binned_statistic(phase, \
                                                  mag - np.median(mag), \
                                                  statistic='median', \
                                                  bins=bins)
        centers = (edges[0:-1] + edges[1:]) / 2.0
        means = np.zeros(n_bins)
        stds = np.zeros(n_bins)
        for j in range(n_bins):
            in_bin = (i_bins - 1 == j)
            if in_bin.any():
                means[j] = np.mean(mag[in_bin] - np.median(mag))
        binned_med_lcs.append(meds)
        binned_mean_lcs.append(means)
        binned_mean_lc_stds.append(stds)
        if rank == 0:
            axes_sum[0].plot(centers, meds, color=cm(feh_cols[i]), alpha=0.4)
            axes_sum[1].plot(centers, means, color=cm(feh_cols[i]), alpha=0.4)

elif dataset == 'crts':

    # dataset settings
    data_dir = 'data/crts_x_sdss/'
    process_raw_lcs = False
    n_bins = 25
    threshold = 3.5

    # get map between CRTS ID and ID number, along with peak time and 
    # period
    css_id = []
    css_id_num = []
    css_period = []
    css_peak = []
    n_skip = 2
    with open(data_dir + 'RRL_params') as f:
        for i, l in enumerate(f):
            if (i > n_skip - 1):
                vals = [val for val in l.split()]
                css_id.append(vals[0])
                css_period.append(float(vals[4]))
                css_peak.append(float(vals[9]))
                css_id_num.append(vals[10])

    # get training stars
    hdulist = apf.open(data_dir + 'crts_bright_feh_info.fit')
    cols = hdulist[1].columns
    data = hdulist[1].data
    ids = data['ID'][0]
    fehs = data['FEH'][0]
    taus = data['PER'][0]
    mus = data['DM'][0]

    # check for bad metallicities
    bad_feh = (fehs < -3.0)
    ids = ids[~bad_feh]
    fehs = fehs[~bad_feh]
    taus = taus[~bad_feh]
    mus = mus[~bad_feh]
    n_lc = len(ids)

    # period distribution
    tau_mean = np.mean(taus)
    tau_std = np.std(taus)

    # plot colours set by metallicities
    feh_min = np.min(fehs)
    feh_max = np.max(fehs)
    feh_cols = (fehs - feh_min) / (feh_max - feh_min)

    # loop through training set
    binned_med_lcs = []
    binned_mean_lcs = []
    binned_mean_lc_stds = []
    if rank == 0:
        fig_sum, axes_sum = mp.subplots(1, 2, figsize=(16,5))
    for i in range(n_lc):
            
        # match IDs
        ind = (j for j,v in enumerate(css_id) if v=='CSS_'+ids[i]).next()

        # build lightcurves and save in simplified format, or read in
        if process_raw_lcs:

            # find matching entries in files
            test = sp.Popen(['/usr/bin/grep ' + css_id_num[ind] + ' ' + \
                             data_dir + '/CSS_RR_phot/*phot'], \
                            shell=True, stdout=sp.PIPE)
            output, err = test.communicate()

            # parse lightcurves
            time = []
            mag = []
            mag_err = []
            lines = output.splitlines()
            for line in lines:
                vals = line.split(',')
                time.append(float(vals[1]))
                mag.append(float(vals[2]))
                mag_err.append(float(vals[3]))
            time = np.array(time)
            mag = np.array(mag)
            mag_err = np.array(mag_err)

            # save to file
            fname = data_dir + css_id[ind] + '_' + css_id_num[ind] + \
                    '_lc.txt'
            np.savetxt(fname, \
                       np.column_stack((time, mag, mag_err)), \
                       fmt='%19.12e', header='time mag mag_err')

        else:
            
            # read lightcurves
            fname = data_dir + css_id[ind] + '_' + css_id_num[ind] + \
                    '_lc.txt'
            lc = np.genfromtxt(fname, names=True)
            time = lc['time']
            mag = lc['mag']
            mag_err = lc['mag_err']

        # what do the phase-wrapped lightcurves look like?
        # 1007116003636; 0.5485033
        period = taus[i]
        phase = np.mod(time - css_peak[ind], period) / period
        if False:
            fig, axes = mp.subplots(1, 2, figsize=(16,5))
            #nu = np.linspace(1.0, 3.0, 1000)
            #power = aps.LombScargle(time, mag, mag_err).power(nu)
            #nu, power = aps.LombScargle(time, mag, mag_err).autopower()
            nu, power = aps.LombScargle(time, mag, mag_err).autopower(minimum_frequency=1.0, maximum_frequency=3.0)
            print nu[np.argmax(power)]
            axes[0].plot(phase, mag, '.', color=cm(feh_cols[i]), alpha=0.4)
            axes[1].plot(nu, power, 'k-')
            axes[1].axvline(1.0 / period, color='r', alpha=0.7, zorder=0)
            mp.suptitle(css_id[ind] + ' / ' + css_id_num[ind])
            #mp.show()

        # calculate some binned statistics
        bins = np.linspace(0, 1, n_bins + 1)
        meds, edges, i_bins = ss.binned_statistic(phase, mag - np.median(mag), \
                                                  statistic='median', \
                                                  bins=bins)
        centers = (edges[0:-1] + edges[1:]) / 2.0
        means = np.zeros(n_bins)
        stds = np.ones(n_bins) * 1e9
        for j in range(n_bins):
            in_bin = (i_bins - 1 == j)
            if in_bin.any():
                stds[j] = np.sqrt(1.0 / np.sum(mag_err[in_bin] ** -2))
                means[j] = np.average(mag[in_bin] - np.median(mag), \
                                      weights=mag_err[in_bin] ** -2)
        binned_med_lcs.append(meds)
        binned_mean_lcs.append(means)
        binned_mean_lc_stds.append(stds)
        if rank == 0:
            axes_sum[0].plot(centers, meds, color=cm(feh_cols[i]), alpha=0.4)
            axes_sum[1].plot(centers, means, color=cm(feh_cols[i]), alpha=0.4)

elif dataset == 'sim':

    def set_phi_13(phi_1, phi_3):
        phi_31 = 2.0 * np.pi + \
                 np.mod(phi_1 - 3.0 * phi_3, np.pi)
        inds = phi_31 > 2.0 * np.pi
        phi_31[inds] = np.pi + \
                       np.mod(phi_1[inds] - 3.0 * phi_3[inds], np.pi)
        return phi_31

    def set_feh(tau, phi_31):
        return -5.038 - 5.394 * tau / 24.0 / 3600.0 + \
               1.345 * phi_31

    # settings
    data_dir = 'data/asas/'
    n_lc = 1000 #10000
    n_fc = 3
    n_samples = 1000
    n_bins = 100

    # stellar properties
    tau_mean = 0.5 * d2s
    tau_std = 0.1 * d2s
    sigma_noise = 0.0 # 0.01

    # stats from arXiv:0906.2199
    raw_stats = np.genfromtxt(data_dir + 'fourier_decomp.txt')
    stats = np.zeros((2 * n_fc, raw_stats.shape[0]))
    for i in range(n_fc):
        stats[i, :] = raw_stats[:, 1 + 2 * i]
        stats[n_fc + i, :] = raw_stats[:, 2 * (i + 1)]
        
        # some stars have negative amplitudes and small phases:
        # shift so they're all in the same quadrant
        weird_phase = stats[n_fc + i, :] < np.pi
        stats[i, weird_phase] *= -1
        stats[n_fc + i, weird_phase] += np.pi
        #mp.plot(stats[i, :], stats[n_fc + i, :], '.')
        #mp.plot(stats[i, weird_phase], stats[n_fc + i, weird_phase], 'r.')
        #mp.show()
    fc_mean = np.mean(stats, 1)
    fc_cov = np.cov(stats)

    # simulate fourier components, periods and metallicities
    fcs = npr.multivariate_normal(fc_mean, fc_cov, n_lc)
    taus = tau_mean + npr.randn(n_lc) * tau_std
    phi_31s = set_phi_13(fcs[:, n_fc + 2], fcs[:, n_fc])
    fehs = set_feh(taus, phi_31s)
    if False:
        mp.plot(phi_31s, fehs, '.')
        phi_31_plot = np.linspace(np.min(phi_31s), np.max(phi_31s))
        mp.plot(phi_31_plot, set_feh(np.mean(taus), phi_31_plot))
        mp.xlabel(r'$\phi_{31}$')
        mp.ylabel(r'${\rm [Fe/H]}$')
        mp.xlim(np.min(phi_31s), np.max(phi_31s))
        mp.ylim(np.min(fehs), np.max(fehs))
        mp.show()

    # plot colours set by metallicities
    feh_min = np.min(fehs)
    feh_max = np.max(fehs)
    feh_cols = (fehs - feh_min) / (feh_max - feh_min)

    # simulate binned lightcurves
    binned_med_lcs = []
    binned_mean_lcs = []
    binned_mean_lc_stds = []
    if rank == 0:
        fig_sum, axes_sum = mp.subplots(1, 2, figsize=(16,5))
    for i in range(n_lc):

        # simulate lightcurves
        phase = npr.rand(n_samples)
        mag = npr.randn(n_samples) * sigma_noise
        for j in range(n_fc):
            mag += fcs[i, j] * np.sin(2.0 * np.pi * (j + 1) * phase + \
                                      fcs[i, n_fc + j])
        #mp.scatter(phase, mag)
        #mp.show()
        
        # calculate some binned statistics
        bins = np.linspace(0, 1, n_bins + 1)
        meds, edges, i_bins = ss.binned_statistic(phase, mag - np.median(mag), \
                                                  statistic='median', \
                                                  bins=bins)
        centers = (edges[0:-1] + edges[1:]) / 2.0
        means = np.zeros(n_bins)
        stds = np.ones(n_bins) * 1e9
        for j in range(n_bins):
            in_bin = (i_bins - 1 == j)
            if in_bin.any():
                stds[j] = sigma_noise / np.sqrt(np.sum(in_bin))
                means[j] = np.mean(mag[in_bin] - np.median(mag))
        binned_med_lcs.append(meds)
        binned_mean_lcs.append(means)
        binned_mean_lc_stds.append(stds)
        if n_lc < 1000 and rank == 0:
            axes_sum[0].plot(centers, meds, color=cm(feh_cols[i]), alpha=0.4)
            axes_sum[1].plot(centers, means, color=cm(feh_cols[i]), alpha=0.4)


# convert binned stats to useful dtype
binned_med_lcs = np.array(binned_med_lcs)
binned_mean_lcs = np.array(binned_mean_lcs)
binned_mean_lc_stds = np.array(binned_mean_lc_stds)

# summarize over stars to obtain median/mean lc shape
med_lc = np.zeros(n_bins)
mean_lc = np.zeros(n_bins)
std_lc = np.zeros(n_bins)
for i in range(n_bins):
    med_lc[i] = np.nanmedian(binned_med_lcs[:, i])
    mean_lc[i] = np.nanmean(binned_mean_lcs[:, i])
    std_lc[i] = np.nanstd(binned_mean_lcs[:, i])

# check for outliers, taking into account intrinsic scatter in each
# bin
if dataset == 'crts':
    is_out = []
    for i in range(n_bins):
        is_out.append(np.abs(mean_lc[i] - binned_mean_lcs[:, i]) / \
                      np.sqrt(std_lc[i] ** 2 + \
                                binned_mean_lc_stds[:, i] ** 2) > threshold)
    out_count = np.sum(np.array(is_out), 0)
    is_out = (out_count > 0)
    for i in range(n_lc):
        if out_count[i] > 0 and rank == 0:
            print 'reject CSS_' + ids[i]
            axes_sum[0].plot(centers, binned_med_lcs[i], 'k')
            axes_sum[1].plot(centers, binned_mean_lcs[i], 'k')
else:
    is_out = np.zeros(n_lc, dtype='bool')

# finish off overlays of lightcurves
if rank == 0:
    if n_lc < 1000:
        axes_sum[0].set_xlabel('phase')
        axes_sum[0].set_ylabel('median mag')
        axes_sum[0].set_ylim(-1.2, 0.6)
        axes_sum[1].set_xlabel('phase')
        axes_sum[1].set_ylabel('iv-weighted mean mag')
        axes_sum[1].set_ylim(-1.2, 0.6)
        fig_sum.savefig(dataset + '_mean_median_phase-wrapped_mags.pdf', \
                        bbox_inches='tight')
        #mp.show()
    mp.close()

# recalculate clean median and mean lightcurves
med_lc_clean = np.zeros(n_bins)
mean_lc_clean = np.zeros(n_bins)
std_lc_clean = np.zeros(n_bins)
n_lc_clean = np.zeros(n_bins)
for i in range(n_bins):
    med_lc_clean[i] = np.nanmedian(binned_med_lcs[~is_out, i])
    mean_lc_clean[i] = np.nanmean(binned_mean_lcs[~is_out, i])
    std_lc_clean[i] = np.nanstd(binned_mean_lcs[~is_out, i])
    n_lc_clean[i] = np.count_nonzero(~np.isnan(binned_mean_lcs[~is_out, i]))

# plot median and mean lightcurves, with and without outliers
if rank == 0:
    fig_stats, axes_stats = mp.subplots(1, 2, figsize=(16,5))
    axes_stats[0].plot(centers, med_lc, 'k-', label='median')
    axes_stats[0].plot(centers, med_lc_clean, 'r--', label='median (clean)')
    axes_stats[1].plot(centers, mean_lc, 'k-', label='mean')
    axes_stats[1].plot(centers, mean_lc_clean, 'r--', label='mean (clean)')
    axes_stats[0].set_xlabel('phase')
    axes_stats[0].set_ylabel('median mag')
    axes_stats[0].set_ylim(-0.8, 0.3)
    axes_stats[0].legend(loc='upper left')
    axes_stats[1].set_xlabel('phase')
    axes_stats[1].set_ylabel('mean mag')
    axes_stats[1].set_ylim(-0.8, 0.3)
    axes_stats[1].legend(loc='upper left')
    mp.savefig(dataset + '_mean_median_lightcurves.pdf', \
               bbox_inches='tight')
    mp.close()
    #mp.show()

# divide through median/mean lightcurve, ditching outliers
if n_lc < 1000 and rank == 0:
    feh_sort = np.argsort(fehs)
    fig, axes = mp.subplots(1, 2, figsize=(16,5))
    for j in range(n_lc):
        i = feh_sort[j]
        if not is_out[i]:
            axes[0].plot(centers, (binned_med_lcs[i, :] - med_lc), \
                         color=cm(feh_cols[i]), alpha=0.4)
            axes[1].plot(centers, (binned_mean_lcs[i, :] - mean_lc), \
                         color=cm(feh_cols[i]), alpha=0.4)
    axes[0].set_xlabel('phase')
    axes[0].set_ylabel('mag / med(mag)')
    axes[1].set_xlabel('phase')
    axes[1].set_ylabel('mag / mean(mag)')
    fig.savefig(dataset + '_mean_median_phase-wrapped_scaled_mags.pdf', \
                bbox_inches='tight')
    mp.close()
    #mp.show()

# test out metallicity dependence in bins
n_bins_feh = 3
feh_min = [-10.0, -1.7, -1.1]
feh_max = [-1.7, -1.1, 10.0]
cols = [cm(0.2), cm(0.5), cm(0.8)]
if rank == 0:
    fig, axes = mp.subplots(2, 2, figsize=(16,10))
med_lc_all_feh = med_lc_clean
mean_lc_all_feh = mean_lc_clean
n_lc_all_feh = n_lc_clean
for k in range(n_bins_feh):

    # summarize over stars to obtain median/mean lc shape
    feh_inds = (fehs >= feh_min[k]) & (fehs < feh_max[k]) & \
               ~is_out
    if rank == 0:
        to_fmt = 'bin {:d} has {:5.2f} < Fe/H < {:5.2f}'
        print to_fmt.format(k, np.min(fehs[feh_inds]), \
                            np.max(fehs[feh_inds]))
    med_lc = np.zeros(n_bins)
    mean_lc = np.zeros(n_bins)
    n_lc_bin = np.zeros(n_bins)
    for i in range(n_bins):
        med_lc[i] = np.nanmedian(binned_med_lcs[feh_inds, i])
        mean_lc[i] = np.nanmean(binned_mean_lcs[feh_inds, i])
        n_lc_bin[i] = np.count_nonzero(~np.isnan(binned_mean_lcs[feh_inds, i]))
    if rank == 0:
        label = '${:4.1f} '.format(feh_min[k]) + r'\leq' + \
                ' [Fe/H] < {:4.1f}$'.format(feh_max[k])
        #axes[0, 0].plot(centers, med_lc, color=cols[k], label=label)
        #axes[0, 1].plot(centers, mean_lc, color=cols[k], label=label)
        #axes[1, 0].plot(centers, med_lc - med_lc_all_feh, color=cols[k], label=label)
        #axes[1, 1].plot(centers, mean_lc - mean_lc_all_feh, color=cols[k], label=label)
        axes[0, 0].errorbar(centers, med_lc, std_lc_clean / np.sqrt(n_lc), color=cols[k], label=label)
        axes[0, 1].errorbar(centers, mean_lc, std_lc_clean / np.sqrt(n_lc), color=cols[k], label=label)
        axes[1, 0].errorbar(centers, med_lc - med_lc_all_feh, std_lc_clean * np.sqrt(1.0 / n_lc + 1.0 / n_lc_bin), color=cols[k], label=label)
        axes[1, 1].errorbar(centers, mean_lc - mean_lc_all_feh, std_lc_clean * np.sqrt(1.0 / n_lc + 1.0 / n_lc_bin), color=cols[k], label=label)

if rank == 0:
    axes[0, 0].set_xlabel('phase')
    axes[0, 0].set_ylabel('median mag')
    axes[0, 0].set_ylim(-0.8, 0.3)
    axes[0, 0].legend(loc='upper left', fontsize=12)
    axes[0, 1].set_xlabel('phase')
    axes[0, 1].set_ylabel('mean mag')
    axes[0, 1].set_ylim(-0.8, 0.3)
    axes[0, 1].legend(loc='upper left', fontsize=12)
    axes[1, 0].set_xlabel('phase')
    axes[1, 0].set_ylabel('median mag - median shape')
    axes[1, 0].set_ylim(-0.08, 0.08)
    axes[1, 0].legend(loc='upper center', fontsize=12)
    axes[1, 1].set_xlabel('phase')
    axes[1, 1].set_ylabel('mean mag - mean shape')
    axes[1, 1].set_ylim(-0.08, 0.08)
    axes[1, 1].legend(loc='upper center', fontsize=12)
    mp.savefig(dataset + '_mean_median_lightcurves_feh_dep.pdf', \
               bbox_inches='tight')
    mp.close()
    #mp.show()

'''
blah = np.zeros(n_bins * n_lc)
for i in range(n_lc):
    blah[i * n_bins: (i + 1) * n_bins] = binned_mean_lcs[i, :]# - mean_lc
mp.hist(blah, bins = 20)
mp.show()
exit()
'''

# should we split into training and test sets or not?
if split_to_train:

    # split into training and test sets. i've coded this as 
    # equal splits, but one set will probably always be different
    n_split = int(np.floor(n_lc / float(set_size)))
    if rank == 0:
        print 'splitting into committee of {:d} nets'.format(n_split)
    set_ids = range(n_lc)
    ra.shuffle(set_ids)
    if use_mpi:
        set_ids = mpi.COMM_WORLD.bcast(set_ids, root=0)
    test_ids = np.zeros((n_lc, n_split), dtype=bool)
    test_ids[:, 0] = [i < set_size for i in set_ids]
    for j in range(1, n_split - 1):
        test_ids[:, j] = [i < set_size * (j + 1) and \
                          i >= set_size * j for i in set_ids]
    test_ids[:, -1] = [i >= set_size * (n_split - 1) for i in set_ids]
    for j in range(n_split):
        fmt_str = 'committee {:d}: {:d} training, {:d} testing'
        if rank == 0:
            print fmt_str.format(j + 1, np.sum(~test_ids[:, j]), \
                                 np.sum(test_ids[:, j]))

# define neural network inputs
#nn_inputs = binned_mean_lcs - mean_lc
nn_inputs = binned_mean_lcs[~is_out, :] - mean_lc
feh_mean = np.mean(fehs[~is_out])
feh_std = np.std(fehs[~is_out])
nn_outputs = (fehs[~is_out] - feh_mean) / feh_std
n_lc = np.sum(~is_out)
if std_by_bin:
    for i in range(n_bins):
        avg_nn_inputs = np.mean(nn_inputs[:, i])
        std_nn_inputs = np.std(nn_inputs[:, i])
        nn_inputs[:, i] = (nn_inputs[:, i] - avg_nn_inputs) / std_nn_inputs
else:
    avg_nn_inputs = np.mean(nn_inputs.flatten())
    std_nn_inputs = np.std(nn_inputs.flatten())
    nn_inputs = (nn_inputs - avg_nn_inputs) / std_nn_inputs
if include_period:
    nn_inputs = np.append(nn_inputs, \
                          (taus[~is_out, None] - tau_mean) / tau_std, 1)

# dependence on training length 
if test_training_length:
    #max_its = np.array([3, 10, 30, 100, 300, 1000, 3000, 10000])
    #max_its = np.array([3, 10, 30, 100, 300, 1000, 3000])
    #max_its = np.array([25, 50, 100, 200, 300, 400, 500])
    max_its = np.array([100, 300, 1000, 3000])
    n_max_its = len(max_its)
    dcol = 1.0 / float(n_max_its - 1)
    seeds = np.random.randint(102314, 221216, n_rpt)
    if use_mpi:
        mpi.COMM_WORLD.Bcast(seeds, root=0)
    feh_pred = np.zeros((n_lc, n_max_its, n_rpt))
    chisq = np.zeros(n_max_its)
    chisq_core = np.zeros(n_max_its)
    feh_pred_loc = np.zeros((n_lc, n_max_its, n_rpt))
    chisq_loc = np.zeros(n_max_its)
    chisq_core_loc = np.zeros(n_max_its)
    n_err_bins = 100
    err_bins = np.linspace(-5.0, 5.0, n_err_bins + 1)
    err_bin_mids = (err_bins[1:] + err_bins[:-1]) / 2.0
    binned_errs = np.zeros((n_err_bins, n_max_its))
    binned_errs_loc = np.zeros((n_err_bins, n_max_its))
    job_list = allocate_jobs_inc_time(n_max_its, n_procs, rank)
    print 'process id {:d}: jobs'.format(rank), job_list
    #for i in range(n_max_its):
    for i in job_list:
        n_lc_core = 0
        for j in range(n_rpt):
            nn = sk.MLPRegressor(hidden_layer_sizes=(20,), \
                                 activation='logistic', solver='lbfgs', \
                                 alpha=0.1, batch_size='auto', \
                                 learning_rate='constant', \
                                 learning_rate_init=0.001, power_t=0.5, \
                                 max_iter=max_its[i], shuffle=True, \
                                 random_state=seeds[j], tol=0.000, \
                                 verbose=False, warm_start=False, \
                                 momentum=0.9, nesterovs_momentum=True, \
                                 early_stopping=False, validation_fraction=0.1, \
                                 beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            for k in range(n_split):
                nn.fit(nn_inputs[~test_ids[:, k], :], \
                       nn_outputs[~test_ids[:, k]])
                feh_pred_loc[test_ids[:, k], i, j] = \
                    nn.predict(nn_inputs[test_ids[:, k], :])
            res = feh_pred_loc[:, i, j] - nn_outputs
            res_rej = np.abs(res) > 1.0
            chisq_loc[i] += np.sum((res) ** 2)
            chisq_core_loc[i] += np.sum((res[~res_rej]) ** 2)
            n_lc_core += np.sum(~res_rej)
            binned_errs_loc[:, i] += np.histogram(res, bins=err_bins)[0]
        chisq_core_loc[i] /= n_lc_core
        print n_lc_core, n_lc
        print 'n_max_its step {:d} of {:d} complete'.format(i + 1, \
                                                            n_max_its)
    chisq_loc /= n_rpt * n_lc
    if use_mpi:
        mpi.COMM_WORLD.barrier()
    chisq = complete_array(chisq_loc, use_mpi)
    chisq_core = complete_array(chisq_core_loc, use_mpi)
    feh_pred = complete_array(feh_pred_loc, use_mpi)
    binned_errs = complete_array(binned_errs_loc, use_mpi)

    # find optimum training length
    opt_ind = np.argmin(chisq)
    opt_ind_core = np.argmin(chisq_core)
    if rank == 0:
        print 'optimum chisq {:f} at {:d}'.format(chisq[opt_ind], \
                                                  max_its[opt_ind])
        print 'or {:f} at {:d} w/out failures'.format(chisq_core[opt_ind_core], \
                                                      max_its[opt_ind_core])

    # plot on main process only
    if rank == 0:

        # plot best performing network
        plot_min = -3.0 * feh_std + feh_mean
        plot_max = 3.0 * feh_std + feh_mean
        mp.plot([plot_min, plot_max], [plot_min, plot_max], 'k')
        mp.scatter(nn_outputs * feh_std + feh_mean, \
                   np.mean(feh_pred[:, opt_ind_core, :], -1) * \
                   feh_std + feh_mean)
        mp.xlabel(r'$[Fe/H]_{\rm true}$')
        mp.ylabel(r'$\langle[Fe/H]_{\rm pred}\rangle$')
        mp.xlim(plot_min, plot_max)
        mp.ylim(plot_min, plot_max)
        mp.savefig(base + '_opt_its_predictions.pdf', \
                   bbox_inches='tight')
        mp.close()

        # plot chi_sq as function of max_its
        mp.semilogx(max_its, chisq, label='all predictions')
        mp.semilogx(max_its, chisq_core, label='failures removed')
        mp.xlabel(r'${\rm n_{its}}$')
        mp.ylabel(r'$\chi^2/{\rm DOF}$')
        mp.xlim(np.min(max_its), np.max(max_its))
        mp.legend(loc='upper right')
        mp.savefig(base + '_max_its_performance.pdf', \
                   bbox_inches='tight')
        mp.close()

        # plot residuals distribution as function of max_its
        res_max = 0.0
        for i in range(n_max_its):
            res_max_temp = np.max(np.abs(err_bin_mids[binned_errs[:, i] > 0]))
            if res_max_temp > res_max:
                res_max = res_max_temp
        n_binned_max = np.max(binned_errs)
        fig, axes = mp.subplots(2, 2, figsize=(16, 5), sharex=True, sharey=True)
        print res_max
        for i in range(n_max_its):
            i_row = i / 2
            i_col = np.mod(i, 2)
            axes[i_row, i_col].step(err_bin_mids, binned_errs[:, i])
            axes[i_row, i_col].text(-0.95 * res_max, 0.9 * n_binned_max, \
                                    '{:d} iterations'.format(max_its[i]))
            axes[i_row, i_col].set_xlim(-res_max, res_max)
            if i_row == 1:
                axes[i_row, i_col].set_xlabel(r'$\Delta[Fe/H]$')
            if i_col == 0:
                axes[i_row, i_col].set_ylabel(r'$N(\Delta[Fe/H])$')
        fig.subplots_adjust(wspace=0, hspace=0)
        mp.savefig(base + '_max_its_residuals.pdf', \
                   bbox_inches='tight')

    if use_mpi:
        mpi.Finalize()
    exit()

# dependence on alpha and n_hidden
n_grid_hid = 5
n_grid_alpha = 5
dcol_hid = 1.0 / float(n_grid_hid - 1)
dcol_alpha = 1.0 / float(n_grid_alpha - 1)
n_hidden = np.linspace(1, 10, n_grid_hid, dtype=int) * 100
n_hidden = np.linspace(1, 10, n_grid_hid, dtype=int) * 2
#alpha = np.logspace(-7, 0, n_grid_alpha)
alpha = np.logspace(-4, 0, n_grid_alpha)
seeds = np.random.randint(102314, 221216, n_rpt)
if use_mpi:
    mpi.COMM_WORLD.Bcast(seeds, root=0)
chisq = np.zeros((n_grid_hid, n_grid_alpha))
chisq_loc = np.zeros((n_grid_hid, n_grid_alpha))
feh_pred = np.zeros((n_lc, n_grid_hid, n_grid_alpha, n_rpt))
feh_pred_loc = np.zeros((n_lc, n_grid_hid, n_grid_alpha, n_rpt))
job_list = allocate_jobs(n_grid_alpha, n_procs, rank)
print 'process id {:d}: jobs'.format(rank), job_list
for i in range(n_grid_hid):
    print 'n_hidden gridpoint {:d}'.format(i + 1)
    for j in job_list:
        print 'alpha gridpoint {:d}'.format(j + 1)
        for k in range(n_rpt):
            #activation='tanh', solver='lbfgs', \
            nn = sk.MLPRegressor(hidden_layer_sizes=(n_hidden[i],), \
                                 activation='logistic', solver='lbfgs', \
                                 alpha=alpha[j], batch_size='auto', \
                                 learning_rate='constant', \
                                 learning_rate_init=0.001, power_t=0.5, \
                                 max_iter=100, shuffle=True, \
                                 random_state=seeds[k], \
                                 tol=0.0, verbose=False, warm_start=False, \
                                 momentum=0.9, nesterovs_momentum=True, \
                                 early_stopping=False, validation_fraction=0.1, \
                                 beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            if split_to_train:
                for m in range(n_split):
                    nn.fit(nn_inputs[~test_ids[:, m], :], \
                           nn_outputs[~test_ids[:, m]])
                    feh_pred_loc[test_ids[:, m], i, j, k] = \
                        nn.predict(nn_inputs[test_ids[:, m], :])
            else:
                nn.fit(nn_inputs, nn_outputs)
                feh_pred_loc[:, i, j, k] = nn.predict(nn_inputs)
            res = feh_pred_loc[:, i, j, k] - nn_outputs
            chisq_loc[i, j] += np.sum(res ** 2)
chisq_loc /= n_rpt * n_lc
if use_mpi:
    mpi.COMM_WORLD.barrier()
chisq = complete_array(chisq_loc, use_mpi)
feh_pred = complete_array(feh_pred_loc, use_mpi)

# find optimum n_hidden and alpha
opt_ind = np.unravel_index(np.argmin(chisq), (n_grid_hid, n_grid_alpha))
if rank == 0:
    print 'optimum chisq {:f} at {:d}, {:e}'.format(chisq[opt_ind], \
                                                    n_hidden[opt_ind[0]], \
                                                    alpha[opt_ind[1]])

# save results to file! but what else to save?
#output_file = open(base + '_opt_alpha_n_hidden_predictions.dat', 'wb')
#feh_pred.tofile(output_file)
#output_file.close()

# no point duplicating plots
if rank == 0:

    # plot best performing network
    plot_min = -3.0 * feh_std + feh_mean
    plot_max = 3.0 * feh_std + feh_mean
    mp.plot([plot_min, plot_max], [plot_min, plot_max], 'k')
    mp.scatter(nn_outputs * feh_std + feh_mean, \
               np.mean(feh_pred[:, opt_ind[0], opt_ind[1], :], -1) * \
               feh_std + feh_mean)
    mp.xlabel(r'$[Fe/H]_{\rm true}$')
    mp.ylabel(r'$\langle[Fe/H]_{\rm pred}\rangle$')
    mp.xlim(plot_min, plot_max)
    mp.ylim(plot_min, plot_max)
    mp.savefig(base + '_opt_alpha_n_hidden_predictions.pdf', \
               bbox_inches='tight')
    mp.close()

    # summary plots
    fig, axes = mp.subplots(1, 2, figsize=(16, 5))
    for i in range(n_grid_hid):
        axes[0].semilogx(alpha, chisq[i, :], color=cm(dcol_hid * i))
    for i in range(n_grid_alpha):
        axes[1].plot(n_hidden, chisq[:, i], color=cm(dcol_alpha * i))
    axes[0].set_xlabel(r'$\alpha$')
    axes[0].set_ylabel(r'$\sum (Z_{\rm pred} - Z_{\rm true})^2$')
    axes[1].set_xlabel(r'$n_{\rm hidden}$')
    axes[1].set_ylabel(r'$\sum (Z_{\rm pred} - Z_{\rm true})^2$')
    mp.savefig(base + '_alpha_n_hidden_1d_performance.pdf', \
               bbox_inches='tight')
    mp.close()

    # 2D plot
    fig = mp.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(chisq, cmap = mpcm.plasma, interpolation = 'nearest')
    mp.colorbar(cax)
    ax.set_xticklabels([''] + ['{:6.1e}'.format(x) for x in alpha])
    ax.set_yticklabels([''] + ['{:d}'.format(x) for x in n_hidden])
    ax.set_xlabel(r'$\alpha$')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel(r'$n_{\rm hidden}$')
    mp.savefig(base + '_alpha_n_hidden_performance.pdf', \
               bbox_inches='tight')
    mp.close()

# @TODO: SAVE TO FILE (networks, eventually)
# @TODO: redo alpha n_hidden with 1000 iterations
#         - then turn on tolerance to see if results same w/ speedup
# @TODO: or loop over max its
# @TODO: increase n_repeat

if use_mpi:
    mpi.Finalize()
