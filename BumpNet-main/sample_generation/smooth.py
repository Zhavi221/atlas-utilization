import numpy as np
import os
import sys
import subprocess
import math
import matplotlib.pyplot as plt
import argparse
import yaml
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.special import gammaln
from scipy import stats
from collections import Counter
    
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import signals
from workspace import Workspace

# GPR Imports
initial_dir = os.getcwd()
from gpr.model import GPmodel, Constraint
from gpr.kern import kernel_RBF, kernel_Matern52
from gpr.preprocessing import LogitScaler, ExtStandardScaler, make_ext_pipeline
os.chdir(initial_dir)

# ROOT Import
import ROOT

from utilities.signatures import get_signature_and_observable
from utilities.data_loading import load_ATLAS

# Define helper functions
def remove_NaN(row): return row[np.isfinite(row)]
def order_of_mag(number): return math.floor(math.log(number, 10))


def load_dark_machines(data_path, verbose=False):
    '''str -> dict
    Loads Dark Machines CSV and ROOT file and extracts histogram names, events, and bin edges.
    Returns a dictionary with keys 'bin_content', 'bin_edges', and 'names'.
    '''

    # Open CSV file
    csv_file = pd.read_csv(data_path)

    # Open ROOT file
    root_file = data_path.replace('csv','root')
    tfile = ROOT.TFile.Open(root_file ,"READ")

    # Split selection and distribution information from the histogram name
    csv_file[['selection', 'distribution']] = csv_file['Hname'].str.split('__', n=1, expand=True)
        
    columns_to_remove = ['Hname', 'nbins', 'ymin', 'ymax', 'selection', 'distribution']
    bin_columns = [c for c in csv_file.columns if c not in columns_to_remove]
    csv_file[bin_columns] = csv_file[bin_columns].apply(pd.to_numeric, errors='coerce')

    # Extract bin_content, bin_edges, and names
    bin_content, names, bin_edges, bin_errors = [], [], [], []
    for i, row in tqdm(csv_file.iterrows(), total=csv_file.shape[0], position=0, leave=True, desc='Loading histograms'):
        
        # Skip massT histograms
        if 'massT' in row['Hname']: continue

        # Get selection and distribution
        selection = row['selection']
        distribution = row['distribution']

        if verbose:
            print(f'> Processing {row["Hname"]}...')

        # Get histogram events
        df_bin_content = row[bin_columns]
        bin_content_i = df_bin_content.to_numpy(dtype=np.int64)

        if verbose:
            print('events', len(bin_content_i), bin_content_i[:3], bin_content_i[-3:])

        # Get bin edges with mass values
        th1 = tfile.Get(row['Hname'])
        if not th1: print("Failed to get histogram\n hist_name = %s\n root_file = %s" % (row['Hname'], root_file))
        th1.SetDirectory(0)
        bin_edges_i = np.array(list(th1.GetXaxis().GetXbins()))
        bin_edges_i = bin_edges_i[:bin_content_i.shape[0]+1]
        
        # Calculate bin errors as the sqrt of the bin content
        bin_errors_i = np.sqrt(bin_content_i)  

        # Ensure there are no duplicates
        bin_content_list = [bc.tolist() for bc in bin_content]
        remove_duplicates = bin_content_i.tolist() not in bin_content_list

        if remove_duplicates:
            bin_content.append(bin_content_i) # Save the DM histogram
            names.append(row['Hname']) # Save the histogram name
            bin_edges.append(bin_edges_i) # Save the mass binning
            bin_errors.append(bin_errors_i) # Save the mass binning

    data = {
        'bin_content': np.array(bin_content, dtype=object),
        'names': np.array(names, dtype=object),
        'bin_edges': np.array(bin_edges, dtype=object),
        'bin_errors': np.array(bin_errors, dtype=object),
    }
    data['bin_content'] = [np.asarray(bc, dtype=np.float64) for bc in data['bin_content']]
    data['bin_edges']   = [np.asarray(be, dtype=np.float64) for be in data['bin_edges']]
    data['bin_errors']  = [np.asarray(be, dtype=np.float64) for be in data['bin_errors']]
    
    tfile.Close()
    print(f'> Successfully loaded {len(data["bin_content"])} histograms from {data_path}')
    return data

def func_forms(bin_content, bin_edges, bin_errors):
    ''' array, array -> array, array, str, str
    Takes as input histogram events and bin edges and fits various functions (polynomial, exponential, gaussian)
    to the data. The best fit is chosen using the AIC (Akaike information criterion) and reduced-chi2 metric. 
    The outputs are the fitted curve (array), Z_LR values (array), fitting function (str),
    and reason for failure (str) if applicable.
    '''

    fail_reason = None

    # Check if bin_content and bin_edges have compatible dimensions
    if len(bin_edges) != len(bin_content) + 1:
        print(f"Error: Length of bin edges {len(bin_edges)} does not match length of histogram {len(bin_content)}")
        fail_reason = 'Length of bin edges does not match length of histogram'
        return None, None, None, fail_reason
    
    # Define functional forms 
    def poly(x, *p): return sum(c*x**i for i, c in enumerate(p))
    def exp(x, *p): return p[0] * np.exp(np.clip(-p[1] * x, -700, 700)) + p[2]
    def gaus(x, *p): return p[0]+p[1]*np.exp(-(x-p[2])**2/(2*p[3])**2)
    
    # Define AIC
    def aic(num_of_params, y_fit, y_true):
        NLL = -np.sum(y_true * np.log(y_fit) - y_fit - gammaln(y_true + 1))
        return 2*num_of_params + 2*NLL
    
    # Define chi2
    def chi2(y_true, y_fit, errors): return np.sum((y_true-y_fit)**2/errors**2)

    # Define fitting function
    def fit(x, y, fit_type, num_of_params, errors=None):
        if 'Polynomial' in fit_type:
            init_params = [0]*(num_of_params + 1)
            if len(init_params) > len(y): raise ValueError
            popt, pcov = curve_fit(poly, x, y, p0=init_params, maxfev=10000, sigma=errors)
            bkg = poly(x, *popt)
        elif fit_type == 'Exponential':
            init_params = [1, 1, lower_bound] if offset > 0 else [1, 1, 1]
            if len(init_params) > len(y): raise ValueError
            popt, pcov = curve_fit(exp, x, y, p0=init_params, maxfev=10000, sigma=errors, 
                                    bounds=([-np.inf,-np.inf,lower_bound],[np.inf,np.inf,np.inf]))
            bkg = exp(x, *popt)
        elif fit_type == 'Gaussian':
            init_params = [lower_bound, 1, 1, 1] if offset > 0 else [1, 1, 1, 1]
            if len(init_params) > len(y): raise ValueError
            popt, pcov = curve_fit(gaus, x, y, p0=init_params, maxfev=10000, sigma=errors, 
                                    bounds=([lower_bound,-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf,np.inf]))
            bkg = gaus(x, *popt)
        chi2_stat = chi2(y, bkg, errors)
        return bkg, popt, chi2_stat
    
    # Fit using bin numbers as the domain
    bin_index = np.arange(len(bin_content))
    
    # Add offset to ensure non-zero uncertainties
    offset = 1-min(bin_content) if min(bin_content) <= 1 else 0
    bin_content = np.asarray(bin_content, dtype=float)
    log_bin_content = np.log(bin_content+offset)

    # Define and propagate errors  
    bin_errors = np.clip(bin_errors, 1e-3, None)
    log_bin_errors = bin_errors/(bin_content+offset) # propagated errors for log transformation
    
    # Specify lower bound to ensure positive events
    lower_bound = np.log(offset) if offset > 0 else -np.inf
    
    # Initialize values
    best_fit = None
    best_fittype = None
    best_ndof = None
    best_chi2 = float('inf')

    # Specify degrees of polynomials 
    degrees_poly = [2, 3, 4, 5]

    # Determine best fit 
    for fit_type in [f"Polynomial_{d}" for d in degrees_poly] + ['Exponential', 'Gaussian']:

        if fit_type==f'Polynomial_{min(degrees_poly)}': aic_nom, degree_nom = None, None
        degree_alt = int(fit_type.split('_')[-1]) if 'Polynomial' in fit_type else 0

        try:
            log_fit, popt, chi2_stat = fit(bin_index, log_bin_content, fit_type, degree_alt, errors=log_bin_errors)
            f = np.exp(log_fit)-offset
        except (RuntimeError, ValueError):
            continue
        
        # Exclude fits with minima
        mins = argrelextrema(f, np.less)
        if (len(mins[0]) >= 1) or (any(y < 0 for y in f)): continue
        
        # Exclude current fit if AIC is not the lowest
        if 'Polynomial' in fit_type:
            aic_alt = aic(degree_alt, f, bin_content) 
            if aic_nom and degree_nom and (aic_alt > aic_nom): continue  
            degree_nom = degree_alt
            aic_nom = aic_alt
        
        # Select fit with the lowest chi2
        if chi2_stat <= best_chi2:
            best_chi2 = chi2_stat
            best_fit = f
            best_ndof = degree_nom if 'Polynomial' in fit_type else 4 if fit_type == 'Gaussian' else 3 if fit_type == 'Exponential' else None
            best_fittype = f'{fit_type.split("_")[0]} (Order: {degree_alt})' if 'Polynomial' in fit_type else fit_type

    bkg = best_fit
    if bkg is None: 
        fail_reason = 'no_valid_fit'
        return None, None, None, fail_reason

    # Calculate Z_LR for saving
    w = Workspace()
    w.W_hypo_bins = 1.0
    w.hypo_sig_func  = getattr(signals, 'gaussian')
    w.data, w.bin_edges, w.bkg_hist = bin_content, bin_edges, bkg
    w.bin_widths = np.diff(bin_edges)
    w.bin_centers = w.bin_edges[:-1] + w.bin_widths/2
    true_z = w.z_scan()

    # Determine goodness-of-fit from p-value of chi2 distribution
    p = stats.chi2.sf(best_chi2, len(bin_content)-best_ndof)
    if p < 0.05:
        fail_reason = 'p_less_0.05'
        return bkg, true_z, best_fittype, fail_reason

    return bkg, true_z, best_fittype, fail_reason


def gpr(bin_content, bin_edges, config, verbose=False):
    '''array, array -> array, array, str, str
    Takes as input histogram events and bin edges and fits data using constrained Gaussian Process Regression (GPR).
    The data is taken through a transformation pipeline then fit using a local GP setup. The outputs are the fitted curve 
    (array), Z_LR values (array), fitting method (str) (which in this case is 'GPR'), and reason for failure (str) if
    relevant.
    '''

    fail_reason = None

    # Check if bin_content and bin_edges have compatible dimensions
    if len(bin_edges) != len(bin_content) + 1:
        print(f"Error: Length of bin edges {len(bin_edges)} does not match length of histogram {len(bin_content)}")
        fail_reason = 'Length of bin edges does not match length of histogram'
        return None, None, None, fail_reason

    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    widths = np.array([bin_centers[i+1]-bin_centers[i] for i in range(0, len(bin_centers)-1)])

    order_of_magnitude = order_of_mag(bin_content[0]) if order_of_mag(bin_content[0]) > 1 else 2

    # Define data transform pipeline
    scaler_x = make_ext_pipeline(LogitScaler(epsilon_0=10**(-order_of_magnitude+1), epsilon_1=1e-1), ExtStandardScaler())
    scaler_y = make_ext_pipeline(LogitScaler(epsilon_0=10**(-order_of_magnitude), epsilon_1=1e-1), ExtStandardScaler())

    # Transform with added zero at the end of data for later calculation of bound constraint
    bin_centers_0_scaled = scaler_x.fit_transform(np.append(bin_centers, bin_centers[-1]+widths[-1]).reshape(-1, 1))
    bin_content_0_scaled = scaler_y.fit_transform(np.append(bin_content, 0).reshape(-1, 1)).ravel() 
    zero_value = bin_content_0_scaled[-1]

    # Remove added zero for training
    bin_centers_scaled = bin_centers_0_scaled[:-1]
    bin_content_scaled = bin_content_0_scaled[:-1]

    # Initialize guesses for parameters
    var_init = np.var(bin_content_scaled)
    lengthscale_init = np.std(bin_centers_scaled)
    likelihood_init = 1

    # Setup kernel and GP model
    ker = kernel_Matern52(variance = var_init, lengthscale = [lengthscale_init])
    model = GPmodel(kernel = ker, likelihood = likelihood_init, mean = np.mean(bin_content_scaled), verbatim=verbose) 

    # Add the training data
    model.X_training = bin_centers_scaled
    model.Y_training = bin_content_scaled

    # Define lengthscale bounds based on largest bin resolution
    widths_scaled = np.array([bin_centers_scaled[i+1]-bin_centers_scaled[i] for i in range(0, len(bin_centers_scaled)-1)])
    lengthscale_bounds = [(max(widths_scaled)[0], 10*lengthscale_init)]

    # Optimize parameters 
    model.optimize(include_constraint = False, fix_likelihood = False, 
                   lengthscale_bounds = lengthscale_bounds, var_bound = (1e-6, 10*var_init), 
                   likelihood_bound = (1e-6, 10*likelihood_init))

    mean, cov = model.calc_posterior_unconstrained(bin_centers_scaled)
    mean = np.array(mean).flatten()

    # If unconstrained GPR fit has minima or negative values, add constraints
    mins = argrelextrema(mean, np.less)
    if (len(mins[0]) >= 1) or (any(x < zero_value for x in mean)):
        # Define helper functions
        def constant_function(val):
            def fun(x): return np.array([val]*x.shape[0])
            return fun

        # Define constraints
        constr_bounded = Constraint(LB = constant_function(zero_value), UB = constant_function(float('Inf')))
        constr_deriv = Constraint(LB = constant_function(-float('Inf')), UB = constant_function(0))
        model.constr_bounded, model.constr_deriv = constr_bounded, [constr_deriv]   
        model.constr_likelihood = 1e-6

        # Find virtual observation points
        df, i_add_pts, pc_min = model.find_XV_subop(bounds = [(min(bin_centers_scaled)[0],max(bin_centers_scaled)[0])], opt_method = 'differential_evolution', p_target = 0.9, i_range = None, max_iterations=50)
    
        # Calculate posterior with constraints
        mean, var, perc, mode, samples, times = model.calc_posterior_constrained(bin_centers_scaled, compute_mode = False, num_samples = 1000, save_samples = 0, resample = True)
        mean = np.array(mean).flatten()

        # Check if constrained GPR fit has minima or negative values
        mins = argrelextrema(mean, np.less)
        if (len(mins[0]) >= 2) or (any(x < zero_value for x in mean)): 
            fail_reason = 'no_valid_fit'
            return None, None, None, fail_reason

    bkg = scaler_y.inverse_transform(mean.reshape(-1,1)).ravel()

    # Determine Z_LR
    w = Workspace()
    w.W_hypo_bins = 1.0
    w.hypo_sig_func  = getattr(signals, 'gaussian')
    w.data, w.bin_edges, w.bkg_hist = bin_content, bin_edges, bkg
    w.bin_widths = np.diff(bin_edges)
    w.bin_centers = w.bin_edges[:-1] + w.bin_widths/2
    true_z = w.z_scan()

    # If Z_LR >= 20, reject fit
    if config["reject_zlr"]:
        if max(abs(true_z)) >= 20: 
            fail_reason = 'zlr_geq_20'
            return bkg, true_z, 'GPR', fail_reason

    return bkg, true_z, 'GPR', fail_reason


def plot_fits_distribution(fits, save_dir):
    '''array, str -> None
    Takes as input a list of fitting functions and plots the distribution of fits used.
    This plot is saved to save_dir.
    '''

    def custom_sort_key(item):
        func = item[0]
        if func == "GPR": return (chr(255), float('inf')) # Ensure "GPR" is always last
        parts = func.split(" (Order: ")
        func_type = parts[0]
        # If there's an order, convert it to an integer; otherwise, set a default order (e.g., 0 or a high value)
        order = int(parts[1].rstrip(")")) if len(parts) > 1 else float('inf')
        return (func_type, order)
    
    fittypes_count = dict(sorted(Counter(fits).items(), key=custom_sort_key))
    fittypes = list(fittypes_count.keys())
    counts = list(fittypes_count.values())

    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(max(10, len(set(fits))*0.5), 8))
    bars = plt.bar(fittypes, counts, color='grey')
    plt.xlabel('Functions')
    plt.tick_params(axis='x', rotation=45)
    plt.ylabel('Counts')
    plt.ylim(bottom=0)
    plt.title(r'Frequency of Fits Used')

    for bar in bars:
        yval = bar.get_height()  # Get the height (value) of each bar (this is the count)
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval)} ({yval/len(fits)*100:.0f}%)', ha='center', va='bottom')

    plt.tight_layout()

    # Save plot
    Path(f'{save_dir}/plots').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{save_dir}/plots/fit_distribution.pdf', dpi=200)
    print(f'> Saved fit distribution plot to {save_dir}/plots/fit_distribution.pdf')
    plt.close()
    

def plot_histogram(hist_name, bin_edges, bin_content, bkg, bin_errors, true_z, fittype, fail_reason, config, save_dir):
    '''str, array, array, array, array, array, str, str, array, str -> None
    Takes as input a histogram name, histogram events, bin edges, bin errors, fitting function/method, and reason for failure.
    Plots the histogram with the smoothed background and Z_LR values, if available. Plots are saved to save_dir.
    '''
    # Initialize parameters
    bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
    num_bins = len(bin_centers)
    num_events = sum(bin_content)
    if "skipped_bins" in config:
        start_bin = int(config["skipped_bins"]*len(bin_content))
    else:
        start_bin = 0
    min_num_bins = config["min_num_bins"]
    min_num_events = config["min_num_events"]
    min_total_events = config["min_total_events"]
       
    # Get signature and observable labels
    signature, observable  = get_signature_and_observable(hist_name)
    xlabel = observable+' [GeV]' if bin_centers is not None else 'Bins'

    # If background is None, set ax to a single axis
    if bkg is None:
        fig, ax = plt.subplots(nrows=1, figsize=(6,6), dpi=200, sharex=True, layout="constrained")
        ax = [ax]
        abs_z_max = None
        ax[0].set_xlabel(xlabel, loc='right')
    else:
        gridspec_kw = {'height_ratios': [3, 1], 'hspace': 0.05}
        fig, ax = plt.subplots(nrows=2, figsize=(6,6), dpi=200, sharex=True, gridspec_kw=gridspec_kw, layout="constrained")
        ax[1].set_xlabel(xlabel, loc='right')


    # Top plot: Histogram (Observed + Background)
    ax[0].errorbar(bin_centers, bin_content[start_bin:start_bin+len(bin_centers)], yerr=bin_errors[start_bin:start_bin+len(bin_centers)], drawstyle='steps-mid', color='black', label='Observed', elinewidth=0.5, capsize=2, zorder=1)
    if bkg is not None: ax[0].plot(bin_centers[start_bin:start_bin+len(bkg)], bkg, '-', color='orange', linewidth=3, label=f'Background ({fittype})', zorder=2)
    if bin_content[0] > 1000: 
        ax[0].set_yscale('log')
        ax[0].set_ylabel('Events (Log)', loc='top')
    ax[0].set_ylabel('Events', loc='top')
    ax[0].set_xlim(bin_centers[0], bin_centers[-1])

    # Create "custom" legend
    legend_text = f'nbins: {len(bin_centers)}\nnevents: {sum(bin_content):.2f}'
    handles, labels = ax[0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color='none', label=legend_text))
    ax[0].legend(handles=handles, loc='upper right', fontsize=8, title=signature)

    # Bottom plot: Z_LR values (if background is available)
    if true_z is not None:
        abs_z_max = round(np.max(np.abs(true_z)), 2)
        abs_z_max_idx = np.argmax(np.abs(true_z))

        # Plot Z_LR values
        ax[1].plot(bin_centers[start_bin:start_bin+len(true_z)], true_z, drawstyle='steps-mid', color='blue', label=r'$Z_{LR}$ (max $|Z|$ = '+ f'{abs_z_max})')
        ax[1].axhline(0, color='black', linestyle='--', linewidth=0.8, zorder=1)
        ax[1].axvline(bin_centers[start_bin:start_bin+len(true_z)][abs_z_max_idx], color='blue', linestyle='--', linewidth=0.5)
        ax[1].set_ylabel(r'$Z$', loc='top')
        ax[1].legend(loc='upper right', fontsize=8)

    # Define color mapping for failure reasons
    fail_color_mapping = {
        "no_valid_fit" : ["No valid fit found!", "red"],
        "blacklist" : ["Histogram blacklisted!", "red"],
        f'less_than_{min_num_bins}_bins' : [r"$\leq 25$ bins", "red"],
        f'less_than_{min_num_bins}_bins_after_event_cut' : [r"$\leq 25$ bins"+f' (after cutting histogram at {min_num_events} events)', "red"],
        f'less_than_{min_total_events}_events' : [r"$\leq 100$ total events", "red"],
        "zlr_geq_20" : [r"$|Z_{LR}| \geq 20$", "red"],
        "p_less_0.05" : [r"$p < 0.05$", "red"],
        None : [r"Passed!", "green"]
    }

    fig.suptitle(fail_color_mapping[fail_reason][0], fontsize=10, color=fail_color_mapping[fail_reason][1], fontweight='bold')

    # Save plots
    new_name = f"{hist_name}__{num_bins}bins__{num_events:.2f}events__Zmax{abs_z_max}__{fail_reason}"
    Path(f'{save_dir}/plots').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{save_dir}/plots/{new_name}.png', dpi=300)
    plt.close()
    

def smooth_histogram(parameters):
    '''str, array, array, int, int -> str, array, array, array, str
    Takes as input a histogram name, histogram events, bin edges, minimum number of bins, 
    and minimum number of events. Applies smoothing to the histogram using either functional forms, GPR, or "both".
    Returns the histogram name, original historam events, bin edges, smoothed background, Z_LR values, 
    fitting method, and reason for failure (if any).
    '''

    hist_name, bin_content, bin_edges, bin_errors, nominal_bin_edges, config, verbose = parameters

    # Initialize values
    bkg, true_z, fittype, fail_reason = None, None, None, None

    if "blacklist" in config.keys() and hist_name in config["blacklist"]:
        fail_reason = 'blacklist'
        return hist_name, bin_content, bin_errors, bin_edges, bkg, true_z, fittype, fail_reason
    
    if nominal_bin_edges is None:
        try:
            # Remove data after last bin with 'min_num_events' events
            low_stats_cut = np.where(bin_content > config["min_num_events"])[0]

            if "skipped_bins" in config: 
                start_bin = int(config["skipped_bins"]*len(bin_content))
            else:
                start_bin = 0

            bin_content = bin_content[start_bin:low_stats_cut[-1]+1]
            bin_edges = bin_edges[start_bin:low_stats_cut[-1]+2]
            bin_errors = bin_errors[start_bin:low_stats_cut[-1]+1]
        except:
            fail_reason = f'less_than_{config["min_num_bins"]}_bins_after_event_cut'
            return hist_name, bin_content, bin_errors, bin_edges, bkg, true_z, fittype, fail_reason
        
        if sum(bin_content) < config['min_total_events']:
            fail_reason = f'less_than_{config["min_total_events"]}_events'
            return hist_name, bin_content, bin_errors, bin_edges, bkg, true_z, fittype, fail_reason
    else:
        # Find common bin edges between current and nominal binning
        common_min_edge = max(bin_edges[0], nominal_bin_edges[0])
        common_max_edge = min(bin_edges[-1], nominal_bin_edges[-1])

        # Ensure the common edges actually exist in both binning schemes
        if common_min_edge not in bin_edges or common_min_edge not in nominal_bin_edges:
            print("WARNING: No common bin edges found between current and nominal binning.")
            exit()
        if common_max_edge not in bin_edges or common_max_edge not in nominal_bin_edges:
            print("WARNING: No common bin edges found between current and nominal binning.")
            exit()

        # Find indices that correspond to the common range
        start_idx = np.searchsorted(bin_edges, common_min_edge, side='left')
        end_idx = np.searchsorted(bin_edges, common_max_edge, side='right')

        # Trim to common range
        bin_content = bin_content[start_idx:end_idx-1]
        bin_edges = bin_edges[start_idx:end_idx]
        bin_errors = bin_errors[start_idx:end_idx-1]

    # Check if we have enough bins/events after trimming
    if len(bin_content) < config["min_num_bins"]:
        fail_reason = f'less_than_{config["min_num_bins"]}_bins'
        return hist_name, bin_content, bin_errors, bin_edges, bkg, true_z, fittype, fail_reason
        
    # Functional forms smoothing
    if config['smooth_method'] == 'func-forms':
        bkg, true_z, fittype, fail_reason = func_forms(bin_content, bin_edges, bin_errors)
        if fail_reason == 'Length of bin edges does not match length of histogram': 
            exit()
        elif fail_reason is not None:
            return hist_name, bin_content, bin_errors, bin_edges, bkg, true_z, fittype, fail_reason
    
    # GPR smoothing
    elif config['smooth_method'] == 'gpr':
        bkg, true_z, fittype, fail_reason = gpr(bin_content, bin_edges, config, verbose=verbose)
        if fail_reason == 'Length of bin edges does not match length of histogram': 
            exit()
        elif fail_reason is not None:
            return hist_name, bin_content, bin_errors, bin_edges, bkg, true_z, fittype, fail_reason

    # Hybrid smoothing
    else:
        # Start with functional forms
        bkg, true_z, fittype, fail_reason = func_forms(bin_content, bin_edges, bin_errors)
        if fail_reason == 'Length of bin edges does not match length of histogram': 
            exit() 
        # If smoothing is insufficient, attempt GPR smoothing
        elif fail_reason is not None:
            bkg, true_z, fittype, fail_reason = gpr(bin_content, bin_edges, config, verbose=verbose)
            if fail_reason is not None: 
                return hist_name, bin_content, bin_errors, bin_edges, bkg, true_z, fittype, fail_reason

    return hist_name, bin_content, bin_errors, bin_edges, bkg, true_z, fittype, fail_reason


def smooth(config):
    '''dict -> None
    Loads parameters from config file and smooths histogram data based on specified smoothing method. 
    Saves smoothed data, along with two plots and a file listing all skipped (unsmoothable) histograms. 
    '''

    # Load config properites
    input_path = config['input_path']
    output_dir = config['output_dir']
    smooth_method = config['smooth_method']
    verbose = config['verbose']

    # Load data
    if input_path.lower().endswith(".root"):
        print(f'> Loading ATLAS samples from {input_path}')
        data = load_ATLAS(input_path, MC=True)
    elif input_path.lower().endswith(".csv"):
        print(f'> Loading Dark Machines samples from {input_path}')
        data = load_dark_machines(input_path, verbose=verbose)
    else:
        raise ValueError(f"The input must be a .csv or .root file, instead got: {input_path}")

    ### --- Load nominal samples if specified --- ###

    data['nominal_bin_edges'] = [None]*len(data['names'])

    if 'nominal_dir' in config.keys():
        nominal_dir = config['nominal_dir']
    else:
        nominal_dir = None

    if nominal_dir is not None:
        print(f'> Loading nominal samples from {nominal_dir}')
        nominal_files = ['names', 'bin_edges']
        nominal_data = {file: np.load(f'{nominal_dir}/{file}.npy', allow_pickle=True).tolist() for file in nominal_files}

        # Match nominal samples to current data
        for i, name in enumerate(nominal_data['names']):
            if name in data['names']:
                nominal_index = list(data['names']).index(name[0])
                data['nominal_bin_edges'][nominal_index] = nominal_data['bin_edges'][i] 

    ### --- Load nominal samples if specified --- ###

    # Check if smoothed .npy files already exist in the output directory
    smoothed_files = ['bin_content', 'smoothed_background', 'bin_edges', 'true_z', 'names', 'fit']
    smoothed_paths = [f'{output_dir}/{file}.npy' for file in smoothed_files]

    # Check if rejected .npy files already exist in the output directory
    rejected_files = ['bin_content', 'smoothed_background', 'bin_edges', 'true_z', 'names', 'fit', 'fail_reason']
    rejected_paths = [f'{output_dir}/rejected/{file}.npy' for file in rejected_files]

    if all(os.path.exists(path) for path in smoothed_paths + rejected_paths):
        print("> Found existing .npy files, importing instead of smoothing ...")
        smoothed = {file: np.load(f'{output_dir}/{file}.npy', allow_pickle=True).tolist() for file in smoothed_files}
        rejected = {file: np.load(f'{output_dir}/rejected/{file}.npy', allow_pickle=True).tolist() for file in rejected_files}
    else:
        # Multiprocessing
        parameters = [(data['names'][i], 
                       data['bin_content'][i] - data['bin_content_bsm'][i],  # Subtract BSM contribution for smoothing
                       data['bin_edges'][i], 
                       data['bin_errors'][i],
                       data['nominal_bin_edges'][i],
                       config,
                       verbose) for i in range(len(data['names']))]

        nproc = subprocess.run(["nproc"], capture_output=True)
        n_cpus = int(nproc.stdout)
        with Pool(n_cpus) as pool:
            results = list(tqdm(pool.imap(smooth_histogram, parameters), desc="Smoothing histograms", total=len(parameters)))

        pool.close() 
        pool.join()

        smoothed = {'background':[], 'names':[], 'bin_content':[], 'bin_errors':[], 'bin_edges':[], 'true_z':[], 'fit':[]}
        rejected = {'background':[], 'names':[], 'bin_content':[], 'bin_errors':[], 'bin_edges':[], 'true_z':[], 'fit':[], 'fail_reason':[]}
        for result in results:
            hist_name, bin_content, bin_errors, bin_edges, bkg, true_z, fittype, fail_reason = result
            if fail_reason is not None:
                rejected['names'].append([hist_name])
                rejected['bin_content'].append(bin_content)
                rejected['bin_errors'].append(bin_errors)
                rejected['bin_edges'].append(bin_edges)
                rejected['background'].append(bkg)
                rejected['true_z'].append(true_z)
                rejected['fit'].append(fittype)
                rejected['fail_reason'].append(fail_reason)
            else:
                smoothed['names'].append([hist_name])
                smoothed['bin_content'].append(bin_content)
                smoothed['bin_errors'].append(bin_errors)
                smoothed['bin_edges'].append(bin_edges)
                smoothed['background'].append(bkg)
                smoothed['true_z'].append(true_z)
                smoothed['fit'].append(fittype)

        os.makedirs(f'{output_dir}/rejected', exist_ok=True)
        
        # Save histogram data (bin_content), smoothed curves (background), bin edges (bin_edges), and histogram names (names)
        for content_name, content in smoothed.items(): np.save(f'{output_dir}/{content_name}.npy', np.array(content, dtype=object))
        for content_name, content in rejected.items(): np.save(f'{output_dir}/rejected/{content_name}.npy', np.array(content, dtype=object))
        print(f'> Saved smoothed curves.')

    # Plot distribution of fits
    plot_fits_distribution(smoothed['fit'], f'{output_dir}')

    # Plot all raw histograms, with smoothed curve if available
    for idx in tqdm(range(len(data['names'])), desc="Saving + Plotting Histograms", total=len(data['names'])):
        hist_name_raw = data['names'][idx]
        bin_content_raw = data['bin_content'][idx]
        bin_edges_raw = data['bin_edges'][idx]
        bin_errors_raw = data['bin_errors'][idx]

        # Check if the histogram was smoothed
        if [hist_name_raw] in smoothed['names']:
            idx_smoothed = smoothed['names'].index([hist_name_raw])
            bkg = smoothed['background'][idx_smoothed]
            true_z = smoothed['true_z'][idx_smoothed]
            fittype = smoothed['fit'][idx_smoothed]
            fail_reason = None

        elif [hist_name_raw] in rejected['names']:
            idx_rejected = rejected['names'].index([hist_name_raw])
            bkg = rejected['background'][idx_rejected]
            true_z = rejected['true_z'][idx_rejected]
            fittype = rejected['fit'][idx_rejected]
            fail_reason = rejected['fail_reason'][idx_rejected]

        else:
            print(f'> Warning: Histogram {hist_name_raw} not found in smoothed or rejected lists.')
            continue

        plot_histogram(hist_name_raw, bin_edges_raw, bin_content_raw, bkg, bin_errors_raw, true_z,
                       fittype, fail_reason, config, f'{output_dir}')
    print(f'> Plotted {len(data["names"])} histograms with smoothed curves if available.')
    
    # copy config to output dir
    config_file = f'{output_dir}/config.yaml'
    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

def main():

    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--config', help='Config file', default='configs/default.yaml')
    arg_parse.add_argument('--input_path', help='input path to the histograms')
    arg_parse.add_argument('--output_dir', help='output directory for smoothed histograms')
    arg_parse.add_argument('--nominal_dir', help='input directory of nominal smoothed histograms', nargs="?", default=None, const=None)

    args = arg_parse.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    config = {**config["smooth"], **config["raw_cuts"]}
    
    if args.input_path is not None:
        config["input_path"] = args.input_path
    
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    
    if args.nominal_dir is not None:
        config["nominal_dir"] = args.nominal_dir

    
    print('>>>>> Starting smoothing hists step...')

    smooth(config)

    print('>>>>> Finished smoothing hists step.')

if __name__ == '__main__':
    main()
