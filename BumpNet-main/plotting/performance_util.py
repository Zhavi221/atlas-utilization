import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colormaps
from scipy.stats import gaussian_kde
from pathlib import Path
from seaborn import heatmap
from pandas import DataFrame
import re
import os
from tqdm import tqdm
import sys


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utilities.signatures import get_signature_and_observable
from utilities.data_loading import load_ATLAS

import text_utils 
globals().update({k: v for k, v in vars(text_utils).items() if '_latex' in k})

class PerformanceUtil:
    def __init__(self, config, input_dir, prediction_dir=None,
                 verbose=False, max_rows=None, seed=1, shuffle=False,
                 edge=(0,1), dpi=100):

        self.dpi = dpi
        self.config = config
        self.cuts = {'min_num_events': config['min_num_events'], 
                    'min_num_bins': config['min_num_bins'],
                    'skipped_bins': config['skipped_bins'],
                    'min_total_events': config['min_total_events']}


        # Initiate random generator
        self.rng = np.random.default_rng(seed)
        
        # check which format we are using (npy recommended for variable number of bins)
        print(f'Loading data from {input_dir}...')
        paths = list(Path(input_dir).rglob(f'*.*'))
        
        # Quick loop to remove unwanted files
        for path in paths:
            input_format = path.suffix.lstrip('.')
            if input_format.endswith("npy")\
            or input_format.endswith("root"):
                break

        # Load files from data samples
        for n in ['bin_content', 'bin_content_bsm', 'true_z', 'background', 'signal_shape', 'wanted_z_and_mu', 'bin_edges', 'names', 'bin_errors']:
            try:
                if verbose: print('starting to load data')
                loaded_data = self.read_data(input_dir, input_format, n, num_rows=max_rows)
            except Exception as e:
                loaded_data = None
                if verbose: print(e)
                print(f'Could not load data for {n}. Setting it to None.')
            setattr(self, n, loaded_data)

        self.n_hists = self.bin_content.shape[0]

        self.zpred = self.read_data(prediction_dir, 'npy', 'pred_z', num_rows=max_rows) if prediction_dir else None
        self.bpred = self.read_data(prediction_dir, 'npy', 'pred_b', num_rows=max_rows) if prediction_dir else None
        self.edge = edge

        # Set attributes

        attribute_names = ['n_bins', 'bin_fraction', 'start_bin', 'last_bin', 'signal_strength', 'wanted_z', 
                           'entries_first_bin', 'bkg_first_bin', 'mass_first_bin',
                           'entries_last_bin', 'bkg_last_bin', 'mass_last_bin',
                           'z_inj', 'entries_sig_bin', 'B_entries_sig_bin', 'mass_sig_bin',
                           'z_pred', 'z_pred_bin', 'z_pred_max', 'z_pred_min', 'z_pred_max_bin', 'z_pred_corrCoeff', 
                           'z_lr', 'z_lr_bin', 'z_lr_max', 'z_lr_min', 'z_lr_max_bin',
                           'entries_pred', 'bkg_pred', 'mass_pred',
                           'entries_lr', 'bkg_lr', 'mass_lr',
                           'entries', 'bkg', 'mass'
                           ]
        attrs = {name:[] for name in attribute_names}
        for i in range(self.n_hists):
            bin_content = self.bin_content[i]
            attrs['n_bins'] += [len(bin_content)]
            
            attrs['entries_first_bin'] += [bin_content[0]]
            attrs['entries_last_bin'] += [bin_content[-1]]
            start_bin = int((edge[0] * len(bin_content))//1)
            last_bin = int((edge[1] * len(bin_content)//1))
            attrs['start_bin'] += [start_bin]
            attrs['last_bin'] += [last_bin]

            attrs['entries'] += [bin_content[start_bin:last_bin]]

            attrs['bin_fraction'] += [np.arange(start_bin, last_bin)/len(bin_content)]

            if self.wanted_z_and_mu is not None:
                attrs['signal_strength'] += [self.wanted_z_and_mu[i,1]]
                attrs['wanted_z'] += [self.wanted_z_and_mu[i,0]]

            if self.background is not None:
                background = self.background[i]
                # in the set of rejected hists from the smoothing, only some of them do not have a background, so we have to do a check and fill with replacement values 
                if background is not None: 
                    attrs['bkg'] += [background[start_bin:last_bin]]
                    attrs['bkg_first_bin'] += [background[0]]
                    attrs['bkg_last_bin'] += [background[-1]]
                else: 
                    attrs['bkg'] += [bin_content[start_bin:last_bin]]
                    attrs['bkg_first_bin'] += [bin_content[0]]
                    attrs['bkg_last_bin'] += [bin_content[-1]]

            mass_edges = np.array(self.bin_edges[i])
            mass = (mass_edges[:-1] + mass_edges[1:])/2
            attrs['mass'] += [mass[start_bin:last_bin]]
            attrs['mass_first_bin'] += [mass[0]]
            attrs['mass_last_bin'] += [mass[-1]]

            if self.signal_shape is not None:
                signal_shape = self.signal_shape[i]
                attrs['z_inj'] += [np.argmax(signal_shape)]
                attrs['entries_sig_bin'] += [bin_content[np.argmax(signal_shape)]]
                attrs['B_entries_sig_bin'] += [background[np.argmax(signal_shape)]]
                attrs['mass_sig_bin'] += [mass[np.argmax(signal_shape)]]

            if self.true_z is not None:
                zlr = self.true_z[i]
                if zlr is not None:
                    attrs['z_lr'] += [zlr[start_bin:last_bin]]
                    attrs['z_lr_bin'] += [np.arange(start_bin, last_bin)]
                    attrs['z_lr_max'] += [np.max(zlr[start_bin:last_bin])]
                    attrs['z_lr_min'] += [np.min(zlr[start_bin:last_bin])]
                    attrs['z_lr_max_bin'] += [start_bin+np.argmax(zlr[start_bin:last_bin])]
                    attrs['entries_lr'] += [bin_content[start_bin+np.argmax(zlr[start_bin:last_bin])]]
                    if background is not None:
                        attrs['bkg_lr'] += [background[start_bin+np.argmax(zlr[start_bin:last_bin])]]
                    attrs['mass_lr'] += [mass[start_bin+np.argmax(zlr[start_bin:last_bin])]]
                else:
                    attrs['z_lr'] += [np.zeros((last_bin-start_bin))]
                    attrs['z_lr_bin'] += [np.arange(start_bin, last_bin)]
                    attrs['z_lr_max'] += [0]
                    attrs['z_lr_min'] += [0]
                    attrs['z_lr_max_bin'] += [0]
                    attrs['entries_lr'] += [0]
                    attrs['bkg_lr'] += [0]
                    attrs['mass_lr'] += [0]

            if self.zpred is not None:
                zpred = self.zpred[i]
                attrs['z_pred'] += [zpred[start_bin:last_bin]]
                attrs['z_pred_bin'] += [np.arange(start_bin, last_bin)]
                attrs['z_pred_max'] += [np.max(zpred[start_bin:last_bin])]
                attrs['z_pred_min'] += [np.min(zpred[start_bin:last_bin])]
                attrs['z_pred_max_bin'] += [start_bin+np.argmax(zpred[start_bin:last_bin])]
                attrs['z_pred_corrCoeff'] += [np.corrcoef(zpred[start_bin:last_bin-1], zpred[start_bin+1:last_bin])[0,1]]
                attrs['entries_pred'] += [bin_content[start_bin+np.argmax(zpred[start_bin:last_bin])]]
                if self.background is not None:
                    attrs['bkg_pred'] += [self.background[i][start_bin+np.argmax(zpred[start_bin:last_bin])]]
                attrs['mass_pred'] += [mass[start_bin+np.argmax(zpred[start_bin:last_bin])]]

        for n in attribute_names:
            if len(attrs[n]) == 0:
                setattr(self, n, None)
            else:
                setattr(self, n, np.array(attrs[n], dtype=object))

        self.cmap =  colormaps["Spectral_r"]
        self.cmap.set_under('white', alpha=0)

        # Shuffle entries
        if shuffle:
            self.shuffle()

        # Flatten attributes that are lists of arrays
        self.flatten(attribute_names)

        print('Finished loading data')

    def shuffle(self):

        print('Shuffling entries...')
        idx = np.arange(0, self.n_hists)
        self.rng.shuffle(idx)
        attrs = ['bin_content', 'bin_content_bsm', 'true_z', 'background', 
                'signal_shape', 'wanted_z_and_mu', 'bin_edges', 'names', 'bin_errors', 'zpred', 'bpred',
                 'n_bins', 'bin_fraction', 'start_bin', 'last_bin', 'signal_strength', 'wanted_z', 
                 'entries_first_bin', 'bkg_first_bin', 'mass_first_bin',
                 'entries_last_bin', 'bkg_last_bin', 'mass_last_bin',
                 'z_inj', 'entries_sig_bin', 'B_entries_sig_bin', 'mass_sig_bin',
                 'z_pred', 'z_pred_bin', 'z_pred_max', 'z_pred_min', 'z_pred_max_bin', 'z_pred_corrCoeff', 
                 'z_lr', 'z_lr_bin', 'z_lr_max', 'z_lr_min', 'z_lr_max_bin',
                 'entries_pred', 'bkg_pred', 'mass_pred',
                 'entries_lr', 'bkg_lr', 'mass_lr',
                 'entries', 'bkg', 'mass']
        for n in attrs:
            old = getattr(self, n)
            if old is None: continue
            setattr(self, n, old[idx])


    def flatten(self, attribute_names):
        for name in attribute_names:
            attr = getattr(self, name)

            if attr is None: 
                continue

            if all(isinstance(v, (list, np.ndarray)) for v in attr): 
                setattr(self, name, np.concatenate(getattr(self, name)))

    
    def read_data(self, dir_path, input_format, which_data='bin_content', num_rows=None):

        if input_format == 'npy':
            path = list(Path(dir_path).glob(f'**/*{which_data}.npy'))[0]
            return np.load(path, allow_pickle=True)

        elif input_format == 'root':
            # Save data dictionary as attribute to avoid repeated loading
            if self.data is None:
                self.data = load_ATLAS(Path(f'{dir_path}/rebinned.root'), cuts=self.cuts)
            return self.data[which_data]
        
        else:
            print(f'Could not load data in appropriate format: {str(input_format)}.')
            return None

        if verbose:
            print(f'Loaded {which_data} from {dir_path}')

        return data

    def get_2d_hist(self, x, y, xlabel, ylabel, title='', ref=None,
                    xlim=None, ylim=None, xlog=None, ylog=None,
                    interpolate=False,
                    x_bins=100, y_bins=100,
                    include_mean=True, include_sd=True, show=False):


        # Get 2D histogram data
        xmin = xlim[0] if xlim is not None else min(x)
        xmax = xlim[1] if xlim is not None else max(x)
        ymin = ylim[0] if ylim is not None else min(y)
        ymax = ylim[1] if ylim is not None else max(y)

        # Start plot
        fig, ax = plt.subplots(1, 1, figsize=(10/2.54, 10/2.54), dpi=self.dpi)

        if interpolate:
            xy = np.vstack([x,y])
            kde = gaussian_kde(xy)
            X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            XY = np.vstack([X.ravel(), Y.ravel()])
            pdf = kde(XY)
            Z = np.reshape(pdf.T, X.shape)
            im = ax.pcolormesh(X, Y, Z, cmap=self.cmap)
        else:
            Z, xedges, yedges, im = ax.hist2d(x, y, bins=(x_bins, y_bins), cmap=self.cmap,
                                    vmin=1e-8, alpha=0.7, range=[[xmin,xmax],[ymin,ymax]])
        if self.config["plot_kde"]:
            from seaborn import kdeplot
            kdeplot(x=x, y=y, ax=ax, fill=False, cmap='Spectral_r',
                    hue_norm=(1e-8, np.max(Z)))

        if ref == 'x':
            ax.plot(np.arange(x.min(),x.max(),0.01),
                    np.arange(x.min(),x.max(),0.01),
                    color='black', linestyle='--', linewidth=0.7)
        elif ref is not None:
            ax.axhline(y=ref, color='black', linestyle='--', linewidth=0.7)

        mean = np.mean(y)
        sd = np.std(y)

        annotation = ''
        if include_mean:
            annotation += f'\n$\\mu = {mean:.2g}$'
        if include_sd:
            annotation += f'\n$\\sigma = {sd:.2g}$'
        ax.annotate(annotation, xy=(0.65, 1.00), xycoords='axes fraction',
                    ha='left', va='top', fontsize=12)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlog:
            ax.set_xscale('log')
        else:
            ax.set_xlim(xmin, xmax)
        if ylog:
            ax.set_yscale('log')
        else:
            ax.set_ylim(ymin, ymax)
        if not xlog:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
        if not ylog:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
        ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)
        ax.tick_params(which='major', axis='both', length=8)
        ax.tick_params(which='minor', axis='both', length=4)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label='Entries')

        if show: plt.show()

        return fig


    def get_mean_std_hist(self, x, y, xlabel, ylabel, title=None, xlim=None, ylim=None, xlog=None, x_bins=100, y_bins=100, show=False):

        # Get 2D histogram data
        xmin = xlim[0] if xlim is not None else min(x)
        xmax = xlim[1] if xlim is not None else max(x)
        ymin = ylim[0] if ylim is not None else min(y)
        ymax = ylim[1] if ylim is not None else max(y)

        # Start plots
        fig_mean, ax_mean = plt.subplots(1, 1, figsize=(10/2.54, 10/2.54), dpi=self.dpi)
        fig_std, ax_std = plt.subplots(1, 1, figsize=(10/2.54, 10/2.54), dpi=self.dpi)

        _, x_edges, _ = np.histogram2d(x, y, bins=(x_bins,y_bins))

        mean_array = []
        std_array = []
        err_array = []
        x_array = []
        dx_array = []

        for i in range(len(x_edges) - 1):
            
            x_selection = np.all([x >= x_edges[i], x < x_edges[i+1]], axis=0)
            
            y_selected = y[x_selection]

            if len(y_selected) > 1:
                mean_array.append(y_selected.mean())
                std_array.append(y_selected.std(ddof=1))
                err_array.append(y_selected.std(ddof=1)/np.sqrt(len(y_selected)))
                x_array.append((x_edges[i+1] + x_edges[i])/2)
                dx_array.append((x_edges[i+1] - x_edges[i])/2)

        mean_array = np.array(mean_array)
        std_array = np.array(std_array)
        err_array = np.array(err_array)
        x_array = np.array(x_array)
        dx_array = np.array(dx_array)
        
        std_max = std_array.max()

        ax_mean.errorbar(x=x_array, y=mean_array, yerr=err_array, xerr=dx_array,fmt=',')

        ax_std.errorbar(x_array, std_array,yerr=None, xerr = dx_array, fmt=',')

        ax_mean.axhline(y=0, color='black', linestyle='--', linewidth=0.7)

        if title:
            ax_mean.set_title(f'$\\mu$ of {title}')
            ax_std.set_title(f'$\\sigma$ of {title}')
        ax_mean.set_xlabel(xlabel)
        ax_std.set_xlabel(xlabel)
        ax_mean.set_ylabel(f'$\\mu$ of {ylabel}')
        ax_std.set_ylabel(f'$\\sigma$ of {ylabel}')
        
        if xlog:
            ax_mean.set_xscale('log')
            ax_std.set_xscale('log')
        else:
            ax_mean.set_xlim(xmin, xmax)
            ax_std.set_xlim(xmin, xmax)
        
        ax_mean.set_ylim(ymin, ymax)
        ax_std.set_ylim(0, 1.2*std_max)

        if not xlog:
            ax_mean.xaxis.set_minor_locator(AutoMinorLocator())
            ax_std.xaxis.set_minor_locator(AutoMinorLocator())
        
        ax_mean.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
        ax_mean.tick_params(axis='y', which='both', direction='in', left=True, right=True)
        ax_mean.tick_params(which='major', axis='both', length=8)
        ax_mean.tick_params(which='minor', axis='both', length=4)

        ax_std.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
        ax_std.tick_params(axis='y', which='both', direction='in', left=True, right=True)
        ax_std.tick_params(which='major', axis='both', length=8)
        ax_std.tick_params(which='minor', axis='both', length=4)

        if show: plt.show()

        return fig_mean, fig_std

    def get_example(self, i, use_bin_centers=True, title=None,
                    z_prediction=True, bkg_prediction=False, show=False, show_corrCoeff=False):
        """
        Plot the observed, bin-by-bin z, background and signal for a sample
        i : index of the histogram being plotted

        """
        start_bin, last_bin = self.start_bin[i], self.last_bin[i]
        bin_content = self.bin_content[i] #observed
        bin_content_bsm = self.bin_content_bsm[i] if self.bin_content_bsm is not None else None
        bin_content_sm = self.bin_content[i] - self.bin_content_bsm[i] if self.bin_content_bsm is not None else None
        bin_errors = self.bin_errors[i] if self.bin_errors is not None else np.sqrt(np.array(bin_content, dtype=float)) #errors
        zlr = self.true_z[i] if self.true_z is not None else None #z_LR
        bkg = self.background[i] if self.background is not None and self.background[i] is not None else None #background
        zpred = self.zpred[i] if self.zpred[i] is not None else None #predicted z
        bpred = self.bpred[i] if self.bpred[i] is not None else None #predicted b
        z_pred_corrCoeff = self.z_pred_corrCoeff[i] if self.z_pred_corrCoeff is not None else None
        signal_shape = self.signal_shape[i] if self.signal_shape is not None else None #signal
        wanted_z = self.wanted_z_and_mu[i,0] if self.wanted_z_and_mu is not None else None #significance (wanted_z) of the signal
        mass_binning = self.bin_edges[i] if self.bin_edges is not None else None
        bin_centers = (mass_binning[:-1]+mass_binning[1:])/2 if (mass_binning is not None and use_bin_centers) else None
        bin_widths = np.diff(mass_binning) if (mass_binning is not None and use_bin_centers) else None
        z_lr_max_bin = self.z_lr_max_bin[i] if self.z_lr_max_bin is not None and self.z_lr_max_bin[i] is not None else None
        z_pred_max_bin = self.z_pred_max_bin[i] if self.z_pred_max_bin is not None else None
        hist_name = self.names[i][0] if title is None else title

        sig = signal_shape if signal_shape is not None else (
            bin_content_bsm if bin_content_bsm is not None else None
        )
        sig_factor = self.wanted_z_and_mu[i,1] if self.wanted_z_and_mu is not None else 1

        signature, observable  = get_signature_and_observable(hist_name)

        # x-position for text denoting edge cuts
        x_text_edge0 = (bin_centers[0] + bin_centers[start_bin-1]) / 2
        x_text_edge1 = (bin_centers[last_bin-1] + bin_centers[-1]) / 2

        nrows = 1
        if sig is not None: nrows += 1
        if zlr is not None or zpred is not None: nrows += 1
        gridspec_kw = {'height_ratios': [2] + [1]*(nrows-1), 'hspace': 0.0} if nrows != 1 else {}
        fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(10/2.54, 10/2.54),
                               dpi=self.dpi, layout = 'constrained',
                               sharex=True, gridspec_kw=gridspec_kw)

        # Main plot
        xlabel = observable+' [GeV]' if bin_centers is not None else 'Bins'
        bin_centers = bin_centers if bin_centers is not None else np.arange(1, len(o)+1) + 0.5
        bin_widths = bin_widths if bin_widths is not None else np.ones(bin_centers.shape[0])

        ax[0].errorbar(bin_centers, bin_content, label='Observed',
                       color='black', linewidth=1,
                       yerr=bin_errors, ecolor='black', elinewidth=0.5, capsize=2, capthick=0.5,
                       drawstyle='steps-mid')

        if bin_content_sm is not None:
            ax[0].errorbar(bin_centers, bin_content_sm, label='SM Events',
                           color='tab:grey', linewidth=1,
                           drawstyle='steps-mid')

        y_min, y_max = ax[0].get_ylim()
        y_text = y_min + 0.01 * (y_max - y_min)
        if self.edge[0] > 0: 
            ax[0].axvspan(bin_centers[0], bin_centers[start_bin-1], alpha=0.3, color='gray')
            ax[0].text(x_text_edge0, y_text, f'{self.edge[0]*100:.0f}%', fontsize=5, color='gray', ha='center', va='bottom')
        if self.edge[1] < 1: 
            ax[0].axvspan(bin_centers[last_bin-1], bin_centers[-1], alpha=0.3, color='gray')
            ax[0].text(x_text_edge1, y_text, f'{self.edge[1]*100:.0f}%', fontsize=5, color='gray', ha='center', va='bottom')

        if bkg is not None:
            ax[0].errorbar(bin_centers, bkg, label='Background', color='tab:orange', linewidth=1)
        if bkg_prediction and bpred is not None:
            ax[0].errorbar(bin_centers, bpred, label='Predicted background', color='tab:red', linewidth=1,
                        drawstyle='steps-mid')
        ax[0].set_ylabel('Entries', loc='top')

        # Inset
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = ax[0].inset_axes([0.63, 0.2, 0.3, 0.3])
        axins.errorbar(bin_centers, bin_content,
                       color='black', linewidth=1,
                       yerr=bin_errors, ecolor='black', elinewidth=0.5, capsize=2, capthick=0.5,
                       drawstyle='steps-mid')
        if z_lr_max_bin is not None:
            axins.axvline(x=bin_centers[z_lr_max_bin], color='tab:blue', linestyle='--', linewidth='1')
        axins.axvline(x=bin_centers[z_pred_max_bin], color='tab:red', linestyle='--', linewidth='1')
        if bkg is not None:
            axins.errorbar(bin_centers, bkg, color='tab:orange', linewidth=1)
        # Inset -- Find the ROI
        b1 = np.max([                 0, z_pred_max_bin-10])
        b2 = np.min([len(bin_content)-1, z_pred_max_bin+10])
        x1 = bin_centers[b1]//1
        x2 = bin_centers[b2]//1
        y1 = np.max([bin_content[z_pred_max_bin], bin_content[b1]])*1.1
        y2 = np.min([bin_content[len(bin_content)-1], bin_content[b2]])*0.9
        y1 = y1//1+1
        y2 = y2//1
        if y2<10:y2=0
        # Inset -- Set zoomed region limits
        axins.set_xlim(x1, x2)
        axins.set_ylim(y2, y1)

        # Inset -- Hide ticks on inset
        axins.set_xticks(np.linspace(x1, x2, 3)) 
        axins.set_yticks(np.linspace(y2, y1, 3))
        axins.tick_params(axis='x', labelsize=6)  # smaller font size for clarity
        axins.tick_params(axis='y', labelsize=6)  # smaller font size for clarity

        # Observed - background plot, if applicable
        if sig is not None:
            if bkg is not None:
                ax[1].bar(bin_centers, bin_content - bkg, width=bin_widths, label='Obs - bkg',
                          color='#d3d3d3', edgecolor='#d3d3d3') 
            ax[1].errorbar(bin_centers, sig * sig_factor, label='Signal',
                            color='tab:green', linewidth=1, drawstyle='steps-mid')
            y_min, y_max = ax[1].get_ylim()
            y_text = y_min + 0.01 * (y_max - y_min)
            if self.edge[0] > 0: 
                ax[1].axvspan(bin_centers[0], bin_centers[start_bin-1], alpha=0.3, color='gray')
                ax[1].text(x_text_edge0, y_text, f'{self.edge[0]*100:.0f}%', fontsize=5, color='gray', ha='center', va='bottom')
            if self.edge[1] < 1: 
                ax[1].axvspan(bin_centers[last_bin-1], bin_centers[-1], alpha=0.3, color='gray', zorder=3)
                ax[1].text(x_text_edge1, y_text, f'{self.edge[1]*100:.0f}%', fontsize=5, color='gray', ha='center', va='bottom')
            ax[1].axhline(y=0, color='black', linewidth=1)
            ymin = 1.5*min(min(bin_content-bkg), min(sig * sig_factor)) if bkg is not None else 1.5*min(sig * sig_factor)
            ymax = 1.5*max(max(bin_content-bkg), max(sig * sig_factor)) if bkg is not None else 1.5*max(sig * sig_factor)
            ax[1].set_ylim(ymin, ymax)
            ax[1].set_ylabel('Entries', loc='top')


        # Significance plot, if applicable
        k = 1 if sig is None and bin_content_bsm is None else 2
        if zlr is not None:
            ax[k].errorbar(bin_centers, zlr,
                           label='LR ' +  z_max_latex +' = ' + f'{np.nanmax(zlr[start_bin:last_bin]):.1f})',
                           color='tab:blue', linewidth=1, drawstyle='steps-mid')
            ax[k].axvline(bin_centers[z_lr_max_bin], color='tab:blue', linestyle='--', linewidth='0.5')
        if z_prediction and zpred is not None:
            ax[k].errorbar(bin_centers, zpred,
                           label='BumpNet ' + z_max_latex + ' = '+f'{np.nanmax(zpred[start_bin:last_bin]):.1f})',
                           color='tab:red', linewidth=1, drawstyle='steps-mid')
            ax[k].axvline(bin_centers[z_pred_max_bin], color='tab:red', linestyle='--', linewidth='0.5')
            if z_pred_corrCoeff is not None and show_corrCoeff:
                ax[k].text(0.8, 0.6, f'Corr coeff={z_pred_corrCoeff:.2g}', transform=ax[k].transAxes, fontsize=5)

        y_min, y_max = ax[k].get_ylim()
        y_text = y_min + 0.01 * (y_max - y_min)
        if self.edge[0] > 0:
            ax[k].axvspan(bin_centers[0], bin_centers[start_bin-1], alpha=0.3, color='gray')
            ax[k].text(x_text_edge0, y_text, f'{self.edge[0]*100:.0f}%', fontsize=5, color='gray', ha='center', va='bottom')
        if self.edge[1] < 1:
            ax[k].axvspan(bin_centers[last_bin-1], bin_centers[-1], alpha=0.3, color='gray')
            ax[k].text(x_text_edge1, y_text, f'{self.edge[1]*100:.0f}%', fontsize=5, color='gray', ha='center', va='bottom')

        ax[k].axhline(color='black', linewidth=0.7)
        if zlr is not None and zpred is not None:
            ymin_zt = np.nanmin(zlr)
            ymax_zt = np.nanmax(zlr)
            ymin_zp = np.nanmin(zpred)
            ymax_zp = np.nanmax(zpred)
            factor = 1.5 if max(ymax_zp, ymax_zt) > 5 else 3
            ymin = factor*min([v for v in [ymin_zt, ymin_zp] if v is not None]) # supposes minimum is negative!
            ymax = factor*max([v for v in [ymax_zt, ymax_zp] if v is not None])
            ax[k].set_ylim(ymin, ymax)
        ax[k].set_xlabel(xlabel, loc='right')
        ax[k].set_ylabel('Significance', loc='center')

        for k in range(nrows):
            if k == 0:
                ax[k].legend(loc='upper right', ncol=[1,2,2][k], columnspacing=0.1, frameon=False,
                             title=signature, fontsize=8)
            else:
                ax[k].legend(loc='upper right', ncol=[1,2,2][k], columnspacing=0.5, frameon=False,
                             fontsize=8)

        if show: plt.show()

        return fig
    
    def get_systematic_ratio(self, i, nominal, use_bin_centers=True, title=None,
                    z_prediction=True, bkg_prediction=False, show=False):
        """
        Plot the observed, bin-by-bin z, background and signal for a sample
        i : index of the histogram being plotted

        """
        bin_content = self.bin_content[i] #observed
        bin_errors = self.bin_errors[i] if self.bin_errors is not None else np.sqrt(np.array(bin_content, dtype=float)) #errors
        zlr = self.true_z[i] if self.true_z is not None else None #z_LR
        bkg = self.background[i] if self.background is not None else None #background
        zpred = self.zpred[i] if self.zpred is not None else None #predicted z
        bpred = self.bpred[i] if self.bpred is not None else None #predicted b
        sig = self.signal_shape[i] if self.signal_shape is not None else None #signal
        sig_factor = self.wanted_z_and_mu[i,1] if self.wanted_z_and_mu is not None else None #scaling factor (mu) for the signal
        wanted_z = self.wanted_z_and_mu[i,0] if self.wanted_z_and_mu is not None else None #significance (wanted_z) of the signal
        mass_binning = self.bin_edges[i] if self.bin_edges is not None else None
        bin_centers = (mass_binning[:-1]+mass_binning[1:])/2 if (mass_binning is not None and use_bin_centers) else None
        bin_widths = np.diff(mass_binning) if (mass_binning is not None and use_bin_centers) else None
        hist_name = self.names[i] if title is None else title
        z_lr_max_bin = self.z_lr_max_bin[i] if self.z_lr_max_bin is not None else None
        z_pred_max_bin = self.z_pred_max_bin[i] if self.z_pred_max_bin is not None else None

       
        # load everything for nominal as well
        if hist_name in nominal.names:
            j = np.where(nominal.names == hist_name)[0][0]
        else:
            return None
        nominal_bin_content = nominal.bin_content[j] #observed
        nominal_bin_errors = nominal.bin_errors[j] if nominal.bin_errors is not None else np.sqrt(np.array(nominal_bin_content, dtype=float)) #errors
        nominal_zlr = nominal.true_z[j] if nominal.true_z is not None else None #z_LR
        nominal_bkg = nominal.background[j] if nominal.background is not None else None #background
        nominal_zpred = nominal.zpred[j] if nominal.zpred is not None else None #predicted z
        nominal_bpred = nominal.bpred[j] if nominal.bpred is not None else None #predicted b
        nominal_sig = nominal.signal_shape[j] if nominal.signal_shape is not None else None #signal
        nominal_sig_factor = nominal.wanted_z_and_mu[i,1] if nominal.wanted_z_and_mu is not None else None #scaling factor (mu) for the signal
        nominal_wanted_z = nominal.wanted_z_and_mu[i,0] if nominal.wanted_z_and_mu is not None else None #significance (wanted_z) of the signal
        nominal_mass_binning = nominal.bin_edges[j] if nominal.bin_edges is not None else None
        nominal_bin_centers = (nominal_mass_binning[:-1]+nominal_mass_binning[1:])/2 if (nominal_mass_binning is not None and use_bin_centers) else None
        nominal_bin_widths = np.diff(nominal_mass_binning) if (nominal_mass_binning is not None and use_bin_centers) else None
        nominal_hist_name = nominal.names[j] if title is None else title
        nominal_z_lr_max_bin = nominal.z_lr_max_bin[j] if nominal.z_lr_max_bin is not None else None
        nominal_z_pred_max_bin = nominal.z_pred_max_bin[j] if nominal.z_pred_max_bin is not None else None

        signature, observable  = get_signature_and_observable(hist_name)

        nrows = 1
        if sig is not None: nrows += 1
        if zlr is not None or zpred is not None: nrows += 1
        gridspec_kw = {'height_ratios': [2] + [1]*(nrows-1), 'hspace': 0.0} if nrows != 1 else {}
        fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(10/2.54, 10/2.54),
                               dpi=self.dpi, layout = 'constrained',
                               sharex=True, gridspec_kw=gridspec_kw)

        # Main plot
        xlabel = observable+' [GeV]' if bin_centers is not None else 'Bins'
        bin_centers = bin_centers if bin_centers is not None else np.arange(1, len(o)+1) + 0.5
        bin_widths = bin_widths if bin_widths is not None else np.ones(bin_centers.shape[0])

        ax[0].errorbar(bin_centers, bin_content, label='syst Observed',
                color='tab:green', linewidth=1,
                yerr=bin_errors, ecolor='tab:green', elinewidth=0.5, capsize=2, capthick=0.5,
                       drawstyle='steps-mid')
        
        ax[0].errorbar(nominal_bin_centers, nominal_bin_content, label='nominal Observed',
                color='tab:olive', linewidth=1,
                yerr=nominal_bin_errors, ecolor='tab:olive', elinewidth=0.5, capsize=2, capthick=0.5,
                       drawstyle='steps-mid', linestyle='--')

        if bkg is not None:
            ax[0].errorbar(bin_centers, bkg, label='Background', color='tab:orange', linewidth=1)
        if bkg_prediction and bpred is not None:
            ax[0].errorbar(bin_centers, bpred, label='Predicted background', color='tab:red', linewidth=1,
                        drawstyle='steps-mid')
        ax[0].set_ylabel('Entries', loc='top')
        ax[0].set_yscale('log')


        # Observed - background plot, if applicable
        if sig is not None:
            if bkg is not None:
                ax[1].bar(bin_centers, bin_content - bkg, width=bin_widths, label='Obs - bkg',
                          color='#d3d3d3', edgecolor='#d3d3d3') 
            ax[1].errorbar(bin_centers, sig * sig_factor, label='Signal',
                           color='tab:green', linewidth=1, drawstyle='steps-mid')
            ax[1].axhline(y=0, color='black', linewidth=1)
            ymin = 1.5*min( [min(bin_content-bkg)])
            ymax = 1.5*max( [max(bin_content-bkg)])
            ax[1].set_ylim(ymin, ymax)
            ax[1].set_ylabel('Entries', loc='top')


        # Significance plot, if applicable
        k = 1 if sig is None else 2
        if zlr is not None:
            ax[k].errorbar(bin_centers, zlr,
                           label='syst LR ' +  r'(' + f'{np.nanmax(zlr):.1f})',
                           color='tab:blue', linewidth=1, drawstyle='steps-mid')
            ax[k].axvline(bin_centers[np.argmax(zlr)], color='tab:blue', linestyle='--', linewidth='0.5')
            ax[k].errorbar(nominal_bin_centers, nominal_zlr,
                           label='nominal LR ' +  r'(' + f'{np.nanmax(nominal_zlr):.1f})',
                           color='tab:cyan', linewidth=1, drawstyle='steps-mid', linestyle='--')
            ax[k].axvline(nominal_bin_centers[np.argmax(nominal_zlr)], color='tab:cyan', linestyle='--', linewidth='0.5')
        if z_prediction and zpred is not None:
            ax[k].errorbar(bin_centers, zpred,
                           label='syst BumpNet ' + r'('+f'{np.nanmax(zpred):.1f})',
                           color='tab:red', linewidth=1, drawstyle='steps-mid')
            ax[k].axvline(bin_centers[np.argmax(zpred)], color='tab:red', linestyle='--', linewidth='0.5')
            ax[k].errorbar(nominal_bin_centers, nominal_zpred,
                           label='nominal BumpNet ' + r'('+f'{np.nanmax(nominal_zpred):.1f})',
                           color='tab:pink', linewidth=1, drawstyle='steps-mid', linestyle="--")
            ax[k].axvline(nominal_bin_centers[np.argmax(nominal_zpred)], color='tab:pink', linestyle='--', linewidth='0.5')

        ax[k].axhline(color='black', linewidth=0.7)
        if zlr is not None and zpred is not None:
            ymin_zt = np.nanmin(zlr)
            ymax_zt = np.nanmax(zlr)
            ymin_zp = np.nanmin(zpred)
            ymax_zp = np.nanmax(zpred)
            factor = 1.5 if max(ymax_zp, ymax_zt) > 5 else 3
            ymin = factor*min([v for v in [ymin_zt, ymin_zp] if v is not None]) # supposes minimum is negative!
            ymax = factor*max([v for v in [ymax_zt, ymax_zp] if v is not None])
            ax[k].set_ylim(ymin, ymax)
        ax[k].set_xlabel(xlabel, loc='right')
        ax[k].set_ylabel('Significance', loc='center')

        for k in range(nrows):
            if k == 0:
                ax[k].legend(loc='upper right', ncol=[1,2,2][k], columnspacing=0.1, frameon=False,
                             title=signature, fontsize=8)
            else:
                ax[k].legend(loc='upper right', ncol=[1,2,2][k], columnspacing=0.1, frameon=False,
                             fontsize=8)

        if show: plt.show()

        return fig

    def get_scatter(self, x, y, xlabel, ylabel, title='', pos=False, ref=None, text=None):

        # Create figure and axes
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=self.dpi)

        try:
            k = gaussian_kde([x,y]) # Estimate of the probability density function (PDF)
            z = k(np.vstack([x.flatten(), y.flatten()]))
            im = ax.scatter(x, y, s=50, c=z, cmap=self.cmap, vmin=1e-8)
        except:
            im = ax.scatter(x, y, s=50, vmin=1e-8)

        # Plot reference lines
        if ref == 'x':
            ax.plot(np.arange(x.min(),x.max(),0.01),
                    np.arange(x.min(),x.max(),0.01),
                    color='black', linestyle='--', linewidth=0.7)
        elif ref is not None:
            ax.axhline(y=ref, color='black', linestyle='--', linewidth=0.7)

        # Add annotation
        if text:
            ax.annotate(text, xy=(0.75, 0.875), xycoords='axes fraction')

        # Set plot design
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
        ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)
        ax.tick_params(which='major', axis='both', length=12)
        ax.tick_params(which='minor', axis='both', length=8)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        return fig

    def get_predictions_heatmap(self, title='', text='', min_threshold=3, mass_binning=None):

        # Get heatmap data
        df_data = np.array([self.z_pred_max, self.mass_pred]).T
        columns = ['zmax', 'mass']
        df = DataFrame(data=df_data, columns=columns)

        xdata = df[(df['zmax'] >= min_threshold)]['zmax'].to_numpy()
        ydata = df[(df['zmax'] >= min_threshold)]['mass'].to_numpy()

        max_threshold = max(int(np.max(xdata)), min_threshold+12)

        x_binning = np.arange(min_threshold, max_threshold, 1)
        y_binning = mass_binning if mass_binning is not None else np.linspace(min(ydata), max(ydata), 20)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi, layout='constrained')
        heatmap_data, x_binning, y_binning = np.histogram2d(xdata, ydata, bins=[x_binning, y_binning])
        ax = heatmap(heatmap_data.T, annot=True, fmt='.3g',cmap='Blues', cbar_kws={'label': 'Entries'})
        ax.set_xlabel('Predicted ' + z_max_latex)
        ax.set_ylabel('Predicted Mass (GeV)')
        plt.xticks(np.arange(len(x_binning)), [f'{i:.0f}' for i in x_binning], rotation='horizontal')
        plt.yticks(np.arange(len(y_binning)), [f'{i:.0f}' for i in y_binning], rotation='horizontal')

        return fig

    def get_mu_and_std_from_2d_hist(self, x, y, nbin=(100,100)):
        from scipy.stats import norm

        h, xedges, yedges = np.histogram2d(x, y, bins=nbin)
        x_bin_width = xedges[1] - xedges[0]

        x_array = []
        x_slice_mean = []
        x_slice_std = []

        for i in range(xedges.size-1):
            y_vals = y[(x > xedges[i]) & (x <= xedges[i+1])]
            if y_vals.size > 0:
                x_array.append(xedges[i] + x_bin_width/2)
                (mu, sigma) = norm.fit(y_vals)
                x_slice_mean.append(mu)
                x_slice_std.append(sigma)

        x_array = np.array(x_array)
        x_slice_mean = np.array(x_slice_mean)
        x_slice_std = np.array(x_slice_std)

        return x_array, x_slice_mean, x_slice_std

    def get_confidence_intervals(self):

        # Get data
        zlr = self.z_lr_max
        zpred = self.z_pred_max
        dz = zpred - zlr
        x_bins, y_bins = 100, 100

        x, mu, std = self.get_mu_and_std_from_2d_hist(zlr, zpred, (x_bins, y_bins))

        # Create figure
        fig, ax = plt.subplots(figsize=(10/2.54, 10/2.54), dpi=self.dpi, layout='tight')
        ax.errorbar(x, mu, xerr=std*1.96, fmt='.', elinewidth=2, capsize=0,
                    ecolor='tab:blue', color='tab:blue', label='95% CL')
        ax.errorbar(x, mu, xerr=std, fmt='.', elinewidth=2, capsize=0,
                    ecolor='tab:orange', color='tab:orange', label='68% CL')

        ax.legend(loc='upper left', frameon=False)
        ax.set_xlabel(z_true_max_latex, loc='right')
        ax.set_ylabel(z_pred_max_latex, loc='top')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
        ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)
        ax.tick_params(which='major', axis='both', length=8)
        ax.tick_params(which='minor', axis='both', length=4)

        return fig


