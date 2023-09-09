import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern, ConstantKernel as Constant

def gp_regression(df, band, cadence, kernel_form, t_samprate, n_samples, lognorm):
    """
    Performs Gaussian Process Regression on a given light curve, and returns the trained Gaussian Process model along 
    with samples drawn from the posterior, if requested.

    Parameters:
    - df (pd.DataFrame): Input data in the form of a DataFrame. Typically loaded from a JSON file.
    - band (str): Name of the telescope filter.
    - cadence (str): Name of the cadence strategy.
    - kernel_form (str): Specifies the kernel function form for the Gaussian Process. Can be one of ['se', 'rq', 'matern12', 'matern32'].
    - t_samprate (float): Sampling rate in time for sampling from the GP posterior.
    - n_samples (int): Number of samples to draw from the GP posterior. If set to 0, no samples are returned.
    - lognorm (bool): If True, log-transforms the data (useful for data with a log-normal distribution, e.g., flux).

    Returns:
    - gp_model (object): The trained Gaussian Process model.
    - samples_df (pd.DataFrame or None): DataFrame containing samples from the posterior. If n_samples is 0, returns None.

    Notes:
    The alpha parameter is interpreted as the variance of the Gaussian measurement noise.
    """
    
    lc_cad_band = df.loc[cadence, band]
    t, y, yerr = np.array(lc_cad_band['mjd']), np.array(lc_cad_band['y']), np.array(lc_cad_band['yerr'])
    if lognorm:
        yerr = yerr / y
        y = np.log(y)
        
    alpha = (yerr / y) ** 2
    
    # Define kernel function form for modeling the variability (red noise)
    # Note: The simulated light curves do not include white noise
    if kernel_form == 'se':
        kernel = Constant(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e4))
    elif kernel_form == 'rq':
        kernel = Constant(1.0, (1e-3, 1e3)) * RationalQuadratic(1.0, 1.0, (1e-5, 1e4), (1e-5, 1e4))
    elif kernel_form == 'matern12':
        kernel = Constant(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-3, 1e4), 0.5)
    elif kernel_form == 'matern32':
        kernel = Constant(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-3, 1e4), 1.5)
    
    t = np.atleast_2d(t).T
    gp_model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=100, normalize_y=True)
    gp_model.fit(t, y)
    
    if n_samples>0:
        t_sample = np.arange(min(t), max(t)+t_samprate, step=t_samprate)
        t_sample = np.atleast_2d(t_sample).T
        
        gp_samples = gp_model.sample_y(X=t_sample, n_samples=n_samples)
        if lognorm: # Revert the log transformation, if applied; float16 sufficient 
            gp_samples = np.exp(gp_samples).astype(np.float16)

        # Convert results into a final pandas DataFrame
        samples_df = pd.DataFrame(np.hstack((t_sample, gp_samples)), 
                          columns=np.append('Time (MJD)', ['Sample '+str(samp) for samp in np.arange(1, n_samples+1, 1)]))
        
        return gp_model, samples_df
    
    else:
        return gp_model, None


def gpr_on_dataset(lcdir_path, band, cadence, kernel_form, t_samprate, n_samples, lognorm):
    """
    Applies Gaussian Process Regression on multiple light curve realizations for a specified telescope filter and 
    cadence strategy. The function processes each JSON file in the given directory, performs GPR on the data, 
    and records the Akaike Information Criterion (AIC) and samples from the GP posterior.

    Parameters:
    - lcdir_path (str): Path to the directory containing the light curve JSON files.
    - band (str): Name of the telescope filter band for analysis.
    - cadence (str): Name of the cadence strategy for analysis.
    - kernel_form (str): Specifies the kernel function form for the Gaussian Process. 
                         Valid options include ['se', 'rq', 'matern12', 'matern32'].
    - t_samprate (float): Sampling rate in time for sampling from the GP posterior.
    - n_samples (int): Number of samples to draw from the GP posterior. If set to 0, only the GP object is returned.
    - lognorm (bool): If True, applies a log-transform to the data.

    Returns:
    - full_gpreals_df (pd.DataFrame): DataFrame containing AIC values and sample data for each light curve realization.

    Notes:
    The AIC is computed using the negative log marginal likelihood, and is thus helpful for comparing kernel forms.
    """
    
    # Initialize a DataFrame to store AIC values and GP sample data for each realization
    full_gpreals_df = pd.DataFrame(columns=['File Number', 'AIC', 'GP Sample Dict.'])
    
    # Sorting files by their numerical index for processing
    files = sorted(glob(lcdir_path+'/*.json'), key=lambda x: int(x.split('/')[-1].split('.')[0]))[:3]

    for i in tqdm(range(len(files))):
        df = pd.read_json(files[i])
        gp_model, df_samples = gp_regression(df, band, cadence, kernel_form, t_samprate, n_samples, lognorm)

        # Compute the AIC based on the negative log marginal likelihood (useful for kernel comparison)
        nlml = -gp_model.log_marginal_likelihood()
        num_params = len(gp_model.kernel.theta)
        aic = -2 * nlml + 2 * num_params

        # Extract file number to use as an index
        file_number = int(files[i].split('/')[-1].split('.')[0])
        new_row = pd.DataFrame({'File Number': [file_number], 'AIC': [aic], 'GP Sample Dict.': [df_samples.to_dict('list')]})
    
        # Append the new row to the full_gpreals_df DataFrame
        full_gpreals_df = pd.concat([full_gpreals_df, new_row], ignore_index=True)
        
    return full_gpreals_df

