#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import scipy.fftpack
import ast
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[ ]:


def psd(t, y, freq_bins):
    """
    Compute the power spectral density (PSD) for an input time series using Fast Fourier Transform (FFT).
    
    Parameters:
    - t (np.array): Time points in the time series.
    - y (np.array): Signal values of the time series.
    - freq_bins (np.array): Frequency bins for binning the PSD.
    
    Returns:
    - binned_psd (np.array): PSD values binned by the provided frequency bins.
    """
    # Calculate the sampling rate of the uniform GP realizationstime series
    dt = np.diff(t)[0]
    
    # Compute the FFT, consider only positive frequencies
    fft = scipy.fftpack.fft(y)
    freq = scipy.fftpack.fftfreq(len(y), d=dt)
    
    # Compute and bin the normalized PSD 
    norm = 2 * dt / (np.mean(y)**2 * len(y))
    psd = norm * np.abs(fft)**2
    binned_psd, _, _ = binned_statistic(freq[:len(freq)//2], psd[:len(psd)//2], 
                                        statistic='mean', bins=freq_bins)
    
    return binned_psd
    
def psd_across_gpreals(gpreals, freq_bins):
    """
    Compute average PSD for a set of light curves sampled at different times.
    
    Arguments:
    - gpreals (dict): Dictionary containing light curve data.
    - freq_bins (np.array): Desired frequency bins for PSD.
    
    Returns:
    - psd_avg (np.array): Average PSD for all light curves.
    - psd_std (np.array): Standard deviation of the PSDs.
    """ 
    n_samples = len(list(gpreals.keys())) - 1
    psds = np.zeros([n_samples, len(freq_bins)-1])
    
    # Extract shared time values common across realizations
    t = gpreals['Time (MJD)']
    
    # Compute the PSD for each light curve sample
    for sample_idx in range(1, n_samples+1):
        y = gpreals['Sample '+str(sample_idx)]
        psds[sample_idx-1,:] = psd(t, y, freq_bins)
        
    # Compute the average and standard deviation of the PSDs
    psd_avg = np.mean(psds, axis=0)
    psd_std = np.std(psds, axis=0)
    
    return psd_avg, psd_std
        
def fit_psd(freq, psd, psd_err, plot_psd=False):
    """
    Fits a power law model to the PSD and optionally plots the result.
    
    Arguments:
    - freq (np.array): Frequency values.
    - psd (np.array): PSD values.
    - psd_err (np.array): Errors associated with PSD values.
    - plot_psd (bool, optional): Flag to plot the PSD and the fitted model.
    
    Returns:
    - slope (float): Slope of the fitted power law.
    - slope_err (float): Error in the slope.
    """
    def linear(f, a, b):
        return a * f + b

    # Convert to log scale for linear fitting of power law
    log_freq = np.log10(freq)
    log_psd = np.log10(psd)
    log_psd_err = psd_err / (psd * np.log(10))

    # Perform linear fit in log-log scale
    popt, pcov = scipy.optimize.curve_fit(linear, log_freq, log_psd, sigma=log_psd_err)
    perr = np.sqrt(np.diag(pcov))

    # If plotting is enabled, show the PSD with the fitted model
    if plot_psd:
        popt_exp, _ = scipy.optimize.curve_fit(lambda f, b: linear(f, -2, b), log_freq, log_psd, sigma=log_psd_err)
        logf_plot = np.linspace(np.log10(min(freq)), np.log10(max(freq)))
        
        fig, ax = plt.subplots()
        plt.xlabel('Frequency (1/days)', fontsize=14)
        plt.ylabel('Power', fontsize=14)
        ax.tick_params(which='major', length=10)
        ax.tick_params(which='minor', length=5)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.errorbar(freq, psd, yerr=psd_err, fmt='.k', ms=7)
        plt.loglog(10**logf_plot, 10**linear(logf_plot, *popt), c='powderblue', ls='-', lw=2, label='Fit Index: '+str(np.round(popt[0],2)))
        plt.loglog(10**logf_plot, 10**linear(logf_plot, a=-2, b=popt_exp[0]), c='orange', ls='-', lw=2, label='True Index: '+r'$(P\propto f^{-2})$')
        plt.legend(fontsize=12)
        plt.show()
    
    slope, slope_err = popt[0], perr[0]
    return slope, slope_err

def psdslopes_across_dataset(gpreals_path, n_series, freq_bins, band, cadence, kernel_form, plot_psd=True):
    """
    Compute the average PSD of a set of light curves, and returns the frequency bins, 
    combined PSD, and error estimates.
    
    Arguments:
    - gpreals_path (str): Directory containing the light curve data files.
    - n_series (int): Number of the simulated time series to consider in the dataset
    - freq_bins (np.array): Frequency bins for PSD.
    - band (str): Wavelength band of the data.
    - cadence (str): Cadence strategy of the light curve data.
    - kernel_form (str): Kernel function for Gaussian Process regression.
    - plot_psd (bool, optional): Flag to plot the computed PSD.
    
    Returns:
    - psd_avg (np.array): Average PSD for all light curves.
    - psd_std (np.array): Standard deviation of the PSDs.
    - psd_slopes (np.array): Slopes of the power law fits to the PSDs.
    """ 
    freq = np.sqrt(freq_bins[:-1] * freq_bins[1:]) # geometric mean for plotting purposes
    psds = np.zeros([n_series, len(freq_bins)-1])
    psd_slopes = np.zeros([n_series, 2])
    
    # Compute the PSD and fit a power law for each light curve in the dataset
    for idx in tqdm(range(1, n_series+1)):
        # Import only the row of interest
        gpreals_df = pd.read_csv(gpreals_path+'/'+band+'_'+cadence+'_'+kernel_form+'.csv', 
                                 skiprows=lambda x: x != 0 and x != idx)
        gpreals_df['GP Sample Dict.'] = gpreals_df['GP Sample Dict.'].apply(ast.literal_eval)
        
        # Extract sample dictionary to compute the PSD
        gpreals_idx = gpreals_df['GP Sample Dict.'].iloc[0]
        psd_idx, psd_std_idx = psd_across_gpreals(gpreals_idx, freq_bins)
        psds[idx] = psd_idx
        
        # Fit the psd with a power law
        slope, slope_err = fit_psd(freq, psd_idx, psd_std_idx, plot_psd=False)
        psd_slopes[idx] = [slope, slope_err]

    # Compute average and standard deviation of all the PSDs
    psd_avg = np.mean(psds, axis=0)
    psd_std = np.std(psds, axis=0)
    psd_slope_avg = np.mean(psd_slopes, axis=0)
    psd_slope_std = np.std(psd_slopes, axis=0)
        
    if plot_psd:
        _,_ = fit_psd(freq, psd_avg, psd_std, plot_psd=True)
        
    return psd_avg, psd_std, psd_slopes

