import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import sys
import os
import re
from pandas import DataFrame
from pathlib import Path
from performance_util import PerformanceUtil
import plotting_helper as pth
import argparse
import yaml
from scipy.stats import norm
import text_utils 
globals().update({k: v for k, v in vars(text_utils).items() if '_latex' in k})

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utilities.signatures import passes_selection

def area_trapezoidal(x, y):
    """Calculate the area under the function given by the coordinates (x,y),
    using the trapezoidal rule."""

    # Sorting the coordinates according to their x values
    sort_order = np.argsort(x)
    x = np.array(x)[sort_order]
    y = np.array(y)[sort_order]

    # Calculating the area
    area = 0.0
    for i in range(0, len(x)-1):
        area += (x[i+1] - x[i]) * (y[i+1] + y[i]) / 2

    return area

def get_rates(hist_sig, hist_bkg):
    """Calculate true positive rate & false positive rate"""

    tpr_values, fpr_values = [], []
    nbins = hist_sig.shape[0]

    for bin_i in range(nbins):
        tp = np.sum(hist_sig[bin_i:]) # Sensitivity
        fn = np.sum(hist_sig[0:bin_i]) # Type 2 error - Rejecting a correct alt' hypothesis. The right index isn't included
        fp = np.sum(hist_bkg[bin_i:]) # Type 1 error - Rejecting a correct null hypothesis
        tn = np.sum(hist_bkg[0:bin_i]) # Specificity

        tpr = tp/(tp+fn) # True positive rate
        fpr = fp/(fp+tn) # False positive rate = 1 - Specificity = 1 - tn/(tn+fp)

        tpr_values.append(tpr)
        fpr_values.append(fpr)

    return fpr_values, tpr_values

def plot(config):

    # Get general configurations
    what_to_plot = config['what_to_plot']
    conditions = config['conditions']
    dpi = config.get('dpi', 300)
    fmt = config.get('format', 'pdf')
    
    # Get plots for each dataset
    for dataset_config in config['datasets']:

        # Get background performance
        perf_bkg = PerformanceUtil(config,
                                    dataset_config['bkg_input_dir'],
                                    dataset_config['bkg_prediction_dir'],
                                    verbose=config.get('verbose', False),
                                    max_rows=config.get('max_rows', None),
                                    seed=config.get('seed', 1),
                                    shuffle=config.get('shuffle', True),
                                    edge=config.get('edge', (0,1)),
                                    dpi=dpi
                                    ) if 'bkg_input_dir' in dataset_config else None

        # Get S+background performance
        perf_sig = PerformanceUtil(config,
                                    dataset_config['sig_input_dir'],
                                    dataset_config['sig_prediction_dir'],
                                    verbose=config.get('verbose', False),
                                    max_rows=config.get('max_rows', None),
                                    seed=config.get('seed', 1),
                                    shuffle=config.get('shuffle', True),
                                    edge=config.get('edge', (0,1)),
                                    dpi=dpi
                                    ) if 'sig_input_dir' in dataset_config else None
        
        # Get comparison to nominal performance
        perf_nominal = PerformanceUtil(config,
                                    dataset_config['nominal_input_dir'],
                                    dataset_config['nominal_prediction_dir'],
                                    verbose=config.get('verbose', False),
                                    max_rows=config.get('max_rows', None),
                                    seed=config.get('seed', 1),
                                    shuffle=config.get('shuffle', True),
                                    edge=config.get('edge', (0,1)),
                                    dpi=dpi
                                    ) if 'nominal_input_dir' in dataset_config else None

        # Create output and get what to plot
        output_dir = Path(dataset_config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # copy config to output dir
        config_file = f'{output_dir}/config.yaml'
        with open(config_file, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

        # Generate plots
        if ('roc_curve' in what_to_plot or 'all' in what_to_plot) and perf_sig and perf_bkg:
            # Get ROC curve data
            zmax = np.max([np.max(x) for x in [perf_bkg.z_lr_max,
                                               perf_bkg.z_pred_max,
                                               perf_sig.z_lr_max,
                                               perf_sig.z_pred_max]])
            zmin = np.min([np.min(x) for x in [perf_bkg.z_lr_min,
                                               perf_bkg.z_pred_min,
                                               perf_sig.z_lr_min,
                                               perf_sig.z_pred_min]])

            binning = np.linspace(zmin, zmax, num=101, endpoint=True)

            hist_true_bkg, _ = np.histogram(perf_bkg.z_lr_max, bins=binning)
            hist_pred_bkg, _ = np.histogram(perf_bkg.z_pred_max, bins=binning)

            hist_true_sig, _ = np.histogram(perf_sig.z_lr_max, bins=binning)
            hist_pred_sig, _ = np.histogram(perf_sig.z_pred_max, bins=binning)
            
            fpr_true, tpr_true = get_rates(hist_true_sig, hist_true_bkg)
            fpr_pred, tpr_pred = get_rates(hist_pred_sig, hist_pred_bkg)

            auc_true = area_trapezoidal(fpr_true, tpr_true)
            auc_pred = area_trapezoidal(fpr_pred, tpr_pred)

            # Plot ROC curve
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10/2.54, 10/2.54), dpi=dpi)
            ax.scatter(fpr_true, tpr_true, color='tab:blue', label=f'LR (AUC = {auc_true:.3f})',
                    marker='.', linestyle='-')
            ax.scatter(fpr_pred, tpr_pred, color='tab:orange', label=f'BumpNet (AUC = {auc_pred:.3f})',
                    marker='.', linestyle='-')

            xmin = np.min((np.min(fpr_pred), np.min(fpr_true)))
            xmax = np.max((np.max(fpr_pred), np.max(fpr_true)))

            ax.plot(np.linspace(xmin,xmax,int((xmax-xmin)/0.01),endpoint=False), np.linspace(xmin,xmax,int((xmax-xmin)/0.01),endpoint=False),
                    color='black', alpha=0.3, linestyle='--', linewidth=0.7)
            ax.set_ylabel('True positive rate', loc='top')
            ax.set_xlabel('False positive rate', loc='right')
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
            ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)
            ax.tick_params(which='major', axis='both')
            ax.tick_params(which='minor', axis='both')
            ax.legend(loc='lower right', frameon=False)
            if config.get('show',False): plt.show()
            save_path = output_dir/f'roc_curve.{fmt}'
            fig.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {save_path}')


        if ('z_distributions' in what_to_plot or 'all' in what_to_plot):

            hist1 = None
            if perf_sig and perf_sig.z_pred is not None:
                hist1, x1 = pth.get_1d_hist(perf_sig.z_pred)

            hist3 = None
            if perf_sig and perf_sig.z_lr is not None:
                hist3, x3 = pth.get_1d_hist(perf_sig.z_lr)

            hist2 = None
            if perf_bkg and perf_bkg.z_pred is not None:
                hist2, x2 = pth.get_1d_hist(perf_bkg.z_pred)

            hist4 = None
            if perf_bkg and perf_bkg.z_lr is not None:
                hist4, x4 = pth.get_1d_hist(perf_bkg.z_lr)

            if perf_sig and (perf_sig.z_pred is not None) and (perf_sig.z_lr is not None):
                fig = plt.figure(figsize=(10/2.54, 10/2.54), dpi=dpi)
                gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

                # Main plot
                ax_main = fig.add_subplot(gs[0])
                if hist1 is not None:
                    ax_main.errorbar(x1, hist1, label='BumpNet S+B',
                                drawstyle='steps-mid', color='tab:orange', linestyle='-', linewidth=0.7)
                if hist3 is not None:
                    ax_main.fill_between(x3, hist3, label='LR S+B',
                                    step='mid', alpha=0.1, color='tab:orange')

                ax_main.legend(loc='upper right', frameon=False)
                ax_main.set_ylabel('Entries', loc='top')
                ax_main.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

                # Residuals plot
                ax_r = fig.add_subplot(gs[1], sharex=ax_main)

                if hist1 is not None and hist3 is not None:
                    # Interpolate to account for non-identical binning
                    hist3_interp = np.interp(x1, x3, hist3)

                    # Compute residuals
                    residuals = hist1 - hist3_interp
                    ax_r.axhline(1, color='gray', linewidth=0.5)
                    ax_r.step(x1, residuals, where='mid', color='black', linewidth=0.7)
                    ax_r.set_ylabel(r'BumpNet $-$ LR', loc='top')
                    ax_r.set_xlabel(r'$Z$', loc='right')
                    ax_r.xaxis.set_minor_locator(AutoMinorLocator())
                    ax_r.yaxis.set_minor_locator(AutoMinorLocator())
                    ax_r.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
                    ax_r.tick_params(axis='y', which='both', direction='in', left=True, right=True)

                if config.get('show', False): plt.show()
                save_path = output_dir/f'Z_sig.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

            if perf_bkg and (perf_bkg.z_pred is not None) and (perf_bkg.z_lr is not None):
                fig = plt.figure(figsize=(10/2.54, 10/2.54), dpi=dpi)
                gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

                # Main plot
                ax_main = fig.add_subplot(gs[0])
                if hist2 is not None:
                    ax_main.errorbar(x2, hist2, label='BumpNet B',
                                drawstyle='steps-mid', color='tab:blue', linestyle='-', linewidth=0.7)
                if hist4 is not None:
                    ax_main.fill_between(x4, hist4, label='LR B',
                                    step='mid', alpha=0.1, color='tab:blue')
                ax_main.legend(loc='upper right', frameon=False)
                ax_main.set_ylabel('Entries', loc='top')
                ax_main.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

                # Residual plot
                ax_r = fig.add_subplot(gs[1], sharex=ax_main)
                if hist2 is not None and hist4 is not None:

                    # Interpolate to account for non-identical binning
                    hist4_interp = np.interp(x2, x4, hist4)

                    # Compute residuals
                    residuals = hist2 - hist4_interp
                    ax_r.axhline(1, color='gray', linewidth=0.5)
                    ax_r.step(x2, residuals, where='mid', color='black', linewidth=0.7)
                    ax_r.set_ylabel(r'BumpNet $-$ LR', loc='top')
                    ax_r.set_xlabel(r'$Z$', loc='right')
                    ax_r.xaxis.set_minor_locator(AutoMinorLocator())
                    ax_r.yaxis.set_minor_locator(AutoMinorLocator())
                    ax_r.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
                    ax_r.tick_params(axis='y', which='both', direction='in', left=True, right=True)

                if config.get('show', False): plt.show()
                save_path = output_dir/f'Z_bkg.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')


        if ('deltaz_distributions' in what_to_plot or 'all' in what_to_plot) and perf_sig and perf_bkg:
            hist1 = None
            if perf_sig.z_pred is not None and perf_sig.z_lr is not None:
                hist1, x1 = pth.get_1d_hist(perf_sig.z_pred - perf_sig.z_lr)
                mean1 = np.sum(hist1 * x1) / np.sum(hist1)
                std1 = np.sum(hist1 * (x1 - mean1)**2) / np.sum(hist1)
            
            hist2 = None
            if perf_bkg.zpred is not None and perf_bkg.z_lr is not None:
                hist2, x2 = pth.get_1d_hist(perf_bkg.z_pred - perf_bkg.z_lr)
                mean2 = np.sum(hist2 * x2) / np.sum(hist2)
                std2 = np.sum(hist2 * (x2 - mean2)**2) / np.sum(hist2)

            fig, ax = plt.subplots(1, 1, figsize=(10/2.54, 10/2.54), dpi=dpi)
            if hist1 is not None:
                ax.errorbar(x1, hist1,
                    label=f'S+B\n($\mu = ${mean1:.2f}, $\sigma = ${std1:.2g})',
                        drawstyle='steps-mid', color='tab:orange', linestyle='-', linewidth=0.7)
            if hist2 is not None:
                ax.errorbar(x2, hist2,
                    label=f'B\n($\mu = $ {mean2:.2f}, $\sigma = ${std2:.2g})',
                        drawstyle='steps-mid', color='tab:blue', linestyle='-', linewidth=0.7)
            ax.axvline(x=0, linestyle='--', linewidth=0.7, color='black', alpha=0.3)
            ymax = max(max(hist1),max(hist2))
            ax.set_ylim(top=1.25*ymax)
            ax.legend(loc='upper right', frameon=False)
            ax.set_xlabel(delta_z_latex, loc='right')
            ax.set_ylabel('Entries', loc='top')
            ax.set_xlim(config['conditions'].get('deltazmax_lim',None))
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
            ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)
            ax.tick_params(which='major', axis='both')
            ax.tick_params(which='minor', axis='both')
            if config.get('show', False): plt.show()
            save_path = output_dir/f'DeltaZ.{fmt}'
            fig.savefig(str(save_path), bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {save_path}')


        if ('deltaz_vs_zLR' in what_to_plot or 'all' in what_to_plot):
            if perf_sig and perf_sig.z_lr is not None:
                fig = perf_sig.get_2d_hist(
                    perf_sig.z_lr,
                    perf_sig.z_pred - perf_sig.z_lr,
                    z_true_latex,
                    delta_z_latex,
                    ref=0,
                    xlim=(min(perf_sig.z_lr), max(perf_sig.z_lr)),
                    ylim=config['conditions'].get('deltaz_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZ_vs_ZLR_signal.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_sig.get_mean_std_hist(
                        perf_sig.z_lr,
                        perf_sig.z_pred - perf_sig.z_lr,
                        z_true_latex,
                        delta_z_latex,
                        xlim=(min(perf_sig.z_lr), max(perf_sig.z_lr)),
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZ_vs_ZLR_signal_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZ_vs_ZLR_signal_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')


            if perf_bkg and perf_bkg.z_lr is not None:
                fig = perf_bkg.get_2d_hist(
                    perf_bkg.z_lr,
                    perf_bkg.z_pred - perf_bkg.z_lr,
                    z_true_latex,
                    delta_z_latex,
                    ref=0,
                    xlim=(min(perf_bkg.z_lr), max(perf_bkg.z_lr)),
                    ylim=config['conditions'].get('deltaz_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZ_vs_ZLR_background.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_bkg.get_mean_std_hist(
                        perf_bkg.z_lr,
                        perf_bkg.z_pred - perf_bkg.z_lr,
                        z_true_latex,
                        delta_z_latex,
                        xlim=(min(perf_bkg.z_lr), max(perf_bkg.z_lr)),
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZ_vs_ZLR_background_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZ_vs_ZLR_background_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')


        if ('deltaz_vs_zpred' in what_to_plot or 'all' in what_to_plot):
            if perf_sig and perf_sig.z_pred is not None:
                fig = perf_sig.get_2d_hist(
                    perf_sig.z_pred,
                    perf_sig.z_pred - perf_sig.z_lr,
                    z_pred_latex,
                    delta_z_latex,
                    ref=0,
                    xlim=(min(perf_sig.z_pred), max(perf_sig.z_pred)),
                    ylim=config['conditions'].get('deltaz_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZ_vs_Zpred_signal.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_sig.get_mean_std_hist(
                        perf_sig.z_pred,
                        perf_sig.z_pred - perf_sig.z_lr,
                        z_pred_latex,
                        delta_z_latex,
                        xlim=(min(perf_sig.z_pred), max(perf_sig.z_pred)),
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZ_vs_Zpred_signal_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZ_vs_Zpred_signal_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')


            if perf_bkg and perf_bkg.z_pred is not None:
                fig = perf_bkg.get_2d_hist(
                    perf_bkg.z_pred,
                    perf_bkg.z_pred - perf_bkg.z_lr,
                    z_pred_latex,
                    delta_z_latex,
                    ref=0,
                    xlim=(min(perf_bkg.z_pred), max(perf_bkg.z_pred)),
                    ylim=config['conditions'].get('deltaz_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZ_vs_Zpred_background.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_bkg.get_mean_std_hist(
                        perf_bkg.z_pred,
                        perf_bkg.z_pred - perf_bkg.z_lr,
                        z_pred_latex,
                        delta_z_latex,
                        xlim=(min(perf_bkg.z_pred), max(perf_bkg.z_pred)),
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZ_vs_Zpred_background_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZ_vs_Zpred_background_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

        
        if ('deltaz_vs_bin_content' in what_to_plot or 'all' in what_to_plot):
            if perf_sig and perf_sig.z_lr is not None:
                fig = perf_sig.get_2d_hist(
                    perf_sig.entries,
                    perf_sig.z_pred - perf_sig.z_lr,
                    r'Bin Entries',
                    delta_z_latex,
                    ref=0,
                    xlim=(min(perf_sig.entries), max(perf_sig.entries)),
                    ylim=config['conditions'].get('deltaz_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'deltaz_vs_bin_content_signal.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_sig.get_mean_std_hist(
                        perf_sig.entries,
                        perf_sig.z_pred - perf_sig.z_lr,
                        r'Bin Entries',
                        delta_z_latex,
                        xlim=(min(perf_sig.entries), max(perf_sig.entries)),
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'deltaz_vs_bin_content_signal_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'deltaz_vs_bin_content_signal_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

            if perf_bkg and perf_bkg.z_lr is not None:
                fig = perf_bkg.get_2d_hist(
                    perf_bkg.entries,
                    perf_bkg.z_pred - perf_bkg.z_lr,
                    r'Bin Entries',
                    delta_z_latex,
                    ref=0,
                    xlim=(min(perf_bkg.entries), max(perf_sig.entries)),
                    ylim=config['conditions'].get('deltaz_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'deltaz_vs_bin_content_background.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_bkg.get_mean_std_hist(
                        perf_bkg.entries,
                        perf_bkg.z_pred - perf_bkg.z_lr,
                        r'Bin Entries',
                        delta_z_latex,
                        xlim=(min(perf_bkg.entries), max(perf_bkg.entries)),
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'deltaz_vs_bin_content_background_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'deltaz_vs_bin_content_background_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

        if ('deltaz_vs_background' in what_to_plot or 'all' in what_to_plot):
            if perf_sig and perf_sig.z_lr is not None:
                fig = perf_sig.get_2d_hist(
                    perf_sig.bkg,
                    perf_sig.z_pred - perf_sig.z_lr,
                    r'Background',
                    delta_z_latex,
                    ref=0,
                    xlim=(min(perf_sig.bkg), max(perf_sig.bkg)),
                    ylim=config['conditions'].get('deltaz_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'deltaz_vs_background_signal.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_sig.get_mean_std_hist(
                        perf_sig.bkg,
                        perf_sig.z_pred - perf_sig.z_lr,
                        r'Background',
                        delta_z_latex,
                        xlim=(min(perf_sig.bkg), max(perf_sig.bkg)),
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'deltaz_vs_background_signal_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'deltaz_vs_background_signal_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

            if perf_bkg and perf_bkg.z_lr is not None:
                fig = perf_bkg.get_2d_hist(
                    perf_bkg.bkg,
                    perf_bkg.z_pred - perf_bkg.z_lr,
                    r'Background',
                    delta_z_latex,
                    ref=0,
                    xlim=(min(perf_bkg.bkg), max(perf_sig.bkg)),
                    ylim=config['conditions'].get('deltaz_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'deltaz_vs_background_background.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_bkg.get_mean_std_hist(
                        perf_bkg.bkg,
                        perf_bkg.z_pred - perf_bkg.z_lr,
                        r'Background',
                        delta_z_latex,
                        xlim=(min(perf_bkg.bkg), max(perf_bkg.bkg)),
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'deltaz_vs_background_background_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'deltaz_vs_background_background_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

        if ('deltaz_vs_mass' in what_to_plot or 'all' in what_to_plot):
            if perf_sig and perf_sig.z_lr is not None:
                fig = perf_sig.get_2d_hist(
                    perf_sig.mass,
                    perf_sig.z_pred - perf_sig.z_lr,
                    r'Mass',
                    delta_z_latex,
                    ref=0,
                    xlim=(min(perf_sig.mass), max(perf_sig.mass)),
                    ylim=config['conditions'].get('deltaz_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'deltaz_vs_mass_signal.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_sig.get_mean_std_hist(
                        perf_sig.mass,
                        perf_sig.z_pred - perf_sig.z_lr,
                        r'Mass',
                        delta_z_latex,
                        xlim=(min(perf_sig.mass), max(perf_sig.mass)),
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'deltaz_vs_mass_signal_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'deltaz_vs_mass_signal_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

            if perf_bkg and perf_bkg.z_lr is not None:
                fig = perf_bkg.get_2d_hist(
                    perf_bkg.mass,
                    perf_bkg.z_pred - perf_bkg.z_lr,
                    r'Mass',
                    delta_z_latex,
                    ref=0,
                    xlim=(min(perf_bkg.mass), max(perf_sig.mass)),
                    ylim=config['conditions'].get('deltaz_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'deltaz_vs_mass_background.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_bkg.get_mean_std_hist(
                        perf_bkg.mass,
                        perf_bkg.z_pred - perf_bkg.z_lr,
                        r'Mass',
                        delta_z_latex,
                        xlim=(min(perf_bkg.mass), max(perf_bkg.mass)),
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'deltaz_vs_mass_background_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'deltaz_vs_mass_background_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

        if ('deltaz_vs_zLR_bin_ratio' in what_to_plot or 'all' in what_to_plot):
            if perf_sig and perf_sig.z_lr is not None:
                fig = perf_sig.get_2d_hist(
                    perf_sig.bin_fraction,
                    perf_sig.z_pred - perf_sig.z_lr,
                    z_true_latex + ' bin / number of bins',
                    delta_z_latex,
                    ref=0,
                    interpolate=config['conditions'].get('interpolate',None),
                    ylim=config['conditions'].get('deltaz_lim',None),
                    xlim=(0,1),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZ_vs_bin_ratio_signal.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_sig.get_mean_std_hist(
                        perf_sig.bin_fraction,
                        perf_sig.z_pred - perf_sig.z_lr,
                        z_true_latex + ' bin / number of bins',
                        delta_z_latex,
                        xlim=(0,1),
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZ_vs_bin_ratio_signal_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZ_vs_bin_ratio_signal_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

            if perf_bkg and perf_bkg.z_lr is not None:
                fig = perf_bkg.get_2d_hist(
                    perf_bkg.bin_fraction,
                    perf_bkg.z_pred - perf_bkg.z_lr,
                    z_true_latex + ' bin  / number of bins',
                    delta_z_latex,
                    ref=0,
                    interpolate=config['conditions'].get('interpolate',None),
                    ylim=config['conditions'].get('deltaz_lim',None),
                    xlim=(0,1),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZ_vs_bin_ratio_background.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_bkg.get_mean_std_hist(
                        perf_bkg.bin_fraction,
                        perf_bkg.z_pred - perf_bkg.z_lr,
                        z_true_latex + ' bin / number of bins',
                        delta_z_latex,
                        xlim=(0,1),
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZ_vs_bin_ratio_background_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZ_vs_bin_ratio_background_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')


        if ('zmax_distributions' in what_to_plot or 'all' in what_to_plot):
            hist1 = None
            if perf_sig and perf_sig.z_pred_max is not None:
                hist1, x1 = pth.get_1d_hist(perf_sig.z_pred_max)

            hist3 = None
            if perf_sig and perf_sig.z_lr_max is not None:
                hist3, x3 = pth.get_1d_hist(perf_sig.z_lr_max)

            hist2 = None
            if perf_bkg and perf_bkg.z_pred_max is not None:
                hist2, x2 = pth.get_1d_hist(perf_bkg.z_pred_max)

            hist4 = None
            if perf_bkg and perf_bkg.z_lr_max is not None:
                hist4, x4 = pth.get_1d_hist(perf_bkg.z_lr_max)

            # Plot distributions together
            fig = plt.figure(figsize=(10/2.54, 10/2.54), dpi=dpi)
            gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

            # Main plot
            ax_main = fig.add_subplot(gs[0])
            if hist1 is not None:
                ax_main.errorbar(x1, hist1, label='BumpNet S+B',
                            drawstyle='steps-mid', color='tab:orange', linestyle='-', linewidth=0.7)
            if hist3 is not None:
                ax_main.fill_between(x3, hist3, label='LR S+B',
                                step='mid', alpha=0.1, color='tab:orange')
            if hist2 is not None:
                ax_main.errorbar(x2, hist2, label='BumpNet B',
                            drawstyle='steps-mid', color='tab:blue', linestyle='-', linewidth=0.7)
            if hist4 is not None:
                ax_main.fill_between(x4, hist4, label='LR B',
                                step='mid', alpha=0.1, color='tab:blue')
            ax_main.legend(loc='upper right', frameon=False)
            ax_main.set_ylabel('Entries', loc='top')
            ax_main.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

            # Residual plot
            if hist1 is not None and hist2 is not None:
                ax_r = fig.add_subplot(gs[1], sharex=ax_main)

                # Interpolate to account for non-identical binning
                hist3_interp = np.interp(x1, x3, hist3)
                hist4_interp = np.interp(x2, x4, hist4)

                # Compute residuals
                residual_sig = hist1 - hist3_interp
                residual_bkg = hist2 - hist4_interp
                ax_r.axhline(1, color='gray', linewidth=0.5)
                ax_r.step(x1, residual_sig, where='mid', color='tab:orange', linewidth=0.7)
                ax_r.step(x2, residual_bkg, where='mid', color='tab:blue', linewidth=0.7)
                ax_r.set_ylabel(r'BumpNet $-$ LR', loc='top')
                ax_r.set_xlabel(z_max_latex, loc='right')
                ax_r.xaxis.set_minor_locator(AutoMinorLocator())
                ax_r.yaxis.set_minor_locator(AutoMinorLocator())
                ax_r.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
                ax_r.tick_params(axis='y', which='both', direction='in', left=True, right=True)
            else:
                ax_main.set_xlabel(z_max_latex, loc='right')

            if config.get('show', False): plt.show()
            save_path = output_dir/f'Zmax.{fmt}'
            fig.savefig(str(save_path), bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {save_path}')

        if ('z_extremum' in what_to_plot or 'all' in what_to_plot):
            hist1_max, hist1_min = None, None
            if perf_sig and perf_sig.z_pred_max is not None and perf_sig.z_pred_min is not None:
                hist1_max, x1_max = pth.get_1d_hist(np.abs(perf_sig.z_pred_max))
                hist1_min, x1_min = pth.get_1d_hist(np.abs(perf_sig.z_pred_min))

            hist3_max, hist3_min = None, None
            if perf_sig and perf_sig.z_lr_max is not None and perf_sig.z_lr_min is not None:
                hist3_max, x3_max = pth.get_1d_hist(np.abs(perf_sig.z_lr_max))
                hist3_min, x3_min = pth.get_1d_hist(np.abs(perf_sig.z_lr_min))

            hist2_max, hist2_min = None, None
            if perf_bkg and perf_bkg.z_pred_max is not None and perf_bkg.z_pred_min is not None:
                hist2_max, x2_max = pth.get_1d_hist(np.abs(perf_bkg.z_pred_max))
                hist2_min, x2_min = pth.get_1d_hist(np.abs(perf_bkg.z_pred_min))

            hist4_max, hist4_min = None, None
            if perf_bkg and perf_bkg.z_lr_max is not None and perf_bkg.z_lr_min is not None:
                hist4_max, x4_max = pth.get_1d_hist(np.abs(perf_bkg.z_lr_max))
                hist4_min, x4_min = pth.get_1d_hist(np.abs(perf_bkg.z_lr_min))

            # Plot distributions together
            if perf_sig and perf_sig.z_pred_max is not None and perf_sig.z_pred_min is not None:
                fig_sig, ax_sig = plt.subplots(1, 1, figsize=(10/2.54, 10/2.54), dpi=dpi)
                if hist1_max is not None:
                    ax_sig.errorbar(x1_max, hist1_max, label=z_pred_max_latex,
                                drawstyle='steps-mid', color='tab:orange', linestyle='-', linewidth=0.7)
                if hist3_max is not None:
                    ax_sig.fill_between(x3_max, hist3_max, label=z_true_max_latex,
                                    step='mid', alpha=0.4, color='tab:orange')
                if hist1_min is not None:
                    ax_sig.errorbar(x1_min, hist1_min, label=z_pred_min_latex,
                                drawstyle='steps-mid', color='tab:blue', linestyle='-', linewidth=0.7)
                if hist3_min is not None:
                    ax_sig.fill_between(x3_min, hist3_min, label=z_true_min_latex,
                                    step='mid', alpha=0.4, color='tab:blue')
                ax_sig.legend(loc='upper right', frameon=False)
                ax_sig.set_xlabel(z_extremum_latex, loc='right')
                ax_sig.set_ylabel('Entries', loc='top')
                ax_sig.xaxis.set_minor_locator(AutoMinorLocator())
                ax_sig.yaxis.set_minor_locator(AutoMinorLocator())
                ax_sig.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
                ax_sig.tick_params(axis='y', which='both', direction='in', left=True, right=True)
                ax_sig.tick_params(which='major', axis='both')
                ax_sig.tick_params(which='minor', axis='both')
                if config.get('show', False): plt.show()
                save_path = output_dir/f'Zextremum_sig.{fmt}'
                fig_sig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig_sig)
                print(f'Saved {save_path}')
            
            if perf_bkg and perf_bkg.z_pred_max is not None and perf_bkg.z_pred_min is not None:
                fig_bkg, ax_bkg = plt.subplots(1, 1, figsize=(10/2.54, 10/2.54), dpi=dpi)
                if hist2_max is not None:
                    ax_bkg.errorbar(x2_max, hist2_max, label=z_pred_max_latex,
                                drawstyle='steps-mid', color='tab:orange', linestyle='-', linewidth=0.7)
                if hist4_max is not None:
                    ax_bkg.fill_between(x4_max, hist4_max, label=z_true_max_latex,
                                    step='mid', alpha=0.4, color='tab:orange')
                if hist2_min is not None:
                    ax_bkg.errorbar(x2_min, hist2_min, label=z_pred_min_latex,
                                drawstyle='steps-mid', color='tab:blue', linestyle='-', linewidth=0.7)
                if hist4_min is not None:
                    ax_bkg.fill_between(x4_min, hist4_min, label=z_true_min_latex,
                                    step='mid', alpha=0.4, color='tab:blue')
                ax_bkg.legend(loc='upper right', frameon=False)
                ax_bkg.set_xlabel(z_extremum_latex, loc='right')
                ax_bkg.set_ylabel('Entries', loc='top')
                ax_bkg.xaxis.set_minor_locator(AutoMinorLocator())
                ax_bkg.yaxis.set_minor_locator(AutoMinorLocator())
                ax_bkg.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
                ax_bkg.tick_params(axis='y', which='both', direction='in', left=True, right=True)
                ax_bkg.tick_params(which='major', axis='both')
                ax_bkg.tick_params(which='minor', axis='both')
                ax_bkg.set_xlim(0,10)
                if config.get('show', False): plt.show()
                save_path = output_dir/f'Zextremum_bkg.{fmt}'
                fig_bkg.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig_bkg)
                print(f'Saved {save_path}')

        if ('deltazmax_distributions' in what_to_plot or 'all' in what_to_plot) and perf_sig and perf_bkg:
            hist1, x1 = pth.get_1d_hist(perf_sig.z_pred_max - perf_sig.z_lr_max)
            mean1 = np.sum(hist1 * x1) / np.sum(hist1)
            std1 = np.sum(hist1 * (x1 - mean1)**2) / np.sum(hist1)

            hist2, x2 = pth.get_1d_hist(perf_bkg.z_pred_max - perf_bkg.z_lr_max)
            mean2 = np.sum(hist2 * x2) / np.sum(hist2)
            std2 = np.sum(hist2 * (x2 - mean2)**2) / np.sum(hist2)

            fig, ax = plt.subplots(1, 1, figsize=(10/2.54, 10/2.54), dpi=dpi)
            ax.errorbar(x1, hist1,
                        label=f'S+B\n($\mu = ${mean1:.2f}, $\sigma = ${std1:.2g})',
                        drawstyle='steps-mid', color='tab:orange', linestyle='-', linewidth=0.7)
            ax.errorbar(x2, hist2,
                        label=f'B\n($\mu = ${mean2:.2f}, $\sigma = ${std2:.2g})',
                        drawstyle='steps-mid', color='tab:blue', linestyle='-', linewidth=0.7)
            ax.axvline(x=0, linestyle='--', linewidth=0.7, color='black', alpha=0.3)
            ymax = max(max(hist1),max(hist2))
            ax.set_ylim(top=1.25*ymax)
            ax.legend(loc='upper right', frameon=False)
            ax.set_xlabel(delta_zmax_latex, loc='right')
            ax.set_ylabel('Entries', loc='top')
            ax.set_xlim(-5, +5)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
            ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)
            ax.tick_params(which='major', axis='both')
            ax.tick_params(which='minor', axis='both')
            if config.get('show', False): plt.show()
            save_path = output_dir/f'DeltaZmax.{fmt}'
            fig.savefig(str(save_path), bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {save_path}')

        if ('zmax_bin_distributions' in what_to_plot or 'all' in what_to_plot):
            hist1 = None
            if perf_sig and perf_sig.z_pred_max is not None:
                hist1, x1 = pth.get_1d_hist(perf_sig.z_pred_max_bin/perf_sig.n_bins, density=True)

            hist3 = None
            if perf_sig and perf_sig.z_lr_max is not None:
                hist3, x3 = pth.get_1d_hist(perf_sig.z_lr_max_bin/perf_sig.n_bins, density=True)

            hist2 = None
            if perf_bkg and perf_bkg.z_pred_max is not None:
                hist2, x2 = pth.get_1d_hist(perf_bkg.z_pred_max_bin / perf_bkg.n_bins, density=True)

            hist4 = None
            if perf_bkg and perf_bkg.z_lr_max is not None:
                hist4, x4 = pth.get_1d_hist(perf_bkg.z_lr_max_bin / perf_bkg.n_bins, density=True)
            
            hist5 = None
            if perf_bkg and perf_bkg.z_pred_max is not None and perf_bkg.z_lr_max is not None:
                false_discovery = (perf_bkg.z_pred_max > 5) & (perf_bkg.z_lr_max < 5) & (abs(perf_bkg.z_pred_max - perf_bkg.z_lr_max) > 1.5)
                if len(perf_bkg.z_pred_max_bin[false_discovery]) > 1:
                    hist5, x5 = pth.get_1d_hist(perf_bkg.z_pred_max_bin[false_discovery] / perf_bkg.n_bins[false_discovery], density=True)

            fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)
            if hist1 is not None:
                ax.errorbar(x1, hist1, label='BumpNet S+B',
                            drawstyle='steps-mid', color='tab:orange', linestyle='-', linewidth=0.7)
            if hist3 is not None:
                ax.fill_between(x3, hist3, label='LR S+B',
                                step='mid', alpha=0.1, color='tab:orange')
            if hist2 is not None:
                ax.errorbar(x2, hist2, label='BumpNet B',
                            drawstyle='steps-mid', color='tab:blue', linestyle='-', linewidth=0.7)
            if hist4 is not None:
                ax.fill_between(x4, hist4, label='LR B',
                                step='mid', alpha=0.1, color='tab:blue')
            if hist5 is not None:
                ax.errorbar(x5, hist5, label='BumpNet false discoveries',
                            drawstyle='steps-mid', color='tab:pink', linestyle='-', linewidth=0.7)
            ax.legend(loc='upper right', frameon=False)
            ax.set_xlabel(z_max_latex +' bin / Number of bins', loc='right')
            ax.set_ylabel('Entries (%)', loc='top')
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
            ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)
            ax.tick_params(which='major', axis='both')
            ax.tick_params(which='minor', axis='both')
            if config.get('show', False): plt.show()
            save_path = output_dir/f'Zmax_bin.{fmt}'
            fig.savefig(str(save_path), bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {save_path}')

        
        if ('correlation_stat_false_discoveries' in what_to_plot or 'all' in what_to_plot) and perf_bkg.z_pred_max is not None and perf_bkg.z_lr_max is not None and perf_bkg.bkg_first_bin is not None:
            false_discovery = (perf_bkg.z_pred_max > 5) & (perf_bkg.z_lr_max < 5) & (abs(perf_bkg.z_pred_max - perf_bkg.z_lr_max) > 1.5)
            binning = np.linspace(perf_bkg.bkg_first_bin.min(), perf_bkg.bkg_first_bin.max(), 100)

            hist1 = None
            if len(perf_bkg.bkg_first_bin[false_discovery]) > 1:
                hist1, x1 = pth.get_1d_hist(perf_bkg.bkg_first_bin[false_discovery],
                                        density=True, binning=binning)
            hist2 = None
            hist2, x2 = pth.get_1d_hist(perf_bkg.bkg_first_bin,
                                         density=True, binning=binning)
            
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)
            if hist1 is not None:
                ax.errorbar(x1, hist1, label=f'False discoveries ({len(perf_bkg.z_pred_max[false_discovery])})',
                            drawstyle='steps-mid', color='tab:blue', linestyle='-', linewidth=0.7)
            ax.errorbar(x2, hist2, label=f'All ({len(perf_bkg.z_pred_max)})',
                            drawstyle='steps-mid', color='tab:pink', linestyle='-', linewidth=0.7)
            ax.legend(loc='upper right', frameon=False)
            ax.set_xlabel('Statistics in first bin', loc='right')
            ax.set_ylabel('Entries normalised', loc='top')
            ax.set_xscale('log')
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
            ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)
            ax.tick_params(which='major', axis='both')
            ax.tick_params(which='minor', axis='both')
            if config.get('show', False): plt.show()
            save_path = output_dir/f'Statistics.{fmt}'
            fig.savefig(str(save_path), bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {save_path}')

        if ('deltazmax_bin_distributions' in what_to_plot or 'all' in what_to_plot):
            hist1 = None
            if perf_sig is not None and perf_sig.z_pred_max is not None and perf_sig.z_lr_max is not None:
                hist1, x1 = pth.get_1d_hist(perf_sig.z_pred_max_bin - perf_sig.z_lr_max_bin,
                                            density=True)
                mean1 = np.sum(hist1 * x1) / np.sum(hist1)
                std1 = np.sum(hist1 * (x1 - mean1)**2) / np.sum(hist1)
                

            hist2 = None
            if perf_bkg is not None and perf_bkg.z_pred_max is not None and perf_bkg.z_lr_max is not None:
                hist2, x2 = pth.get_1d_hist(perf_bkg.z_pred_max_bin - perf_bkg.z_lr_max_bin,
                                            density=True)
                mean2 = np.sum(hist2 * x2) / np.sum(hist2)
                std2 = np.sum(hist2 * (x2 - mean2)**2) / np.sum(hist2)
                

            fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)

            if hist1 is not None:
                ax.errorbar(x1, hist1,
                            label=f'S+B\n($\mu = ${mean1:.2f}, $\sigma = ${std1:.2g})',
                            drawstyle='steps-mid', color='tab:orange', linestyle='-', linewidth=0.7)
            if hist2 is not None:
                ax.errorbar(x2, hist2,
                            label=f'B\n($\mu = ${mean2:.2f}, $\sigma = ${std2:.2g})',
                            drawstyle='steps-mid', color='tab:blue', linestyle='-', linewidth=0.7)
            ax.axvline(x=0, linestyle='--', linewidth=0.7, color='tab:gray')
            ax.legend(loc='upper right', frameon=False, fontsize=10)
            ax.set_xlabel(delta_zmax_bin_latex, loc='right')
            ax.set_ylabel('Entries (%)', loc='top')
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
            ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)
            ax.tick_params(which='major', axis='both')
            ax.tick_params(which='minor', axis='both')
            if config.get('show', False): plt.show()
            save_path = output_dir/f'DeltaZmax_bin.{fmt}'
            fig.savefig(str(save_path), bbox_inches='tight')

            ax.set_xlim(-60,60)
            ax.set_yscale('log')
            save_path = output_dir/f'DeltaZmax_bin_log.{fmt}'
            fig.savefig(str(save_path), bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {save_path}')

        if ('deltazmax_vs_zLRmax' in what_to_plot or 'all' in what_to_plot):
            if perf_sig and perf_sig.z_lr is not None:
                fig = perf_sig.get_2d_hist(
                    perf_sig.z_lr_max,
                    perf_sig.z_pred_max - perf_sig.z_lr_max,
                    z_true_max_latex,
                    delta_zmax_latex,
                    ref=0,
                    xlim=(0, max(perf_sig.z_lr_max)),
                    ylim=config['conditions'].get('deltazmax_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_vs_ZLRmax_signal.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_sig.get_mean_std_hist(
                        perf_sig.z_lr_max,
                        perf_sig.z_pred_max - perf_sig.z_lr_max,
                        z_true_max_latex,
                        delta_zmax_latex,
                        xlim=(0, max(perf_sig.z_lr_max)),
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZmax_vs_ZLRmax_signal_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZmax_vs_ZLRmax_signal_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

            if perf_bkg and perf_bkg.z_lr is not None:
                fig = perf_bkg.get_2d_hist(
                    perf_bkg.z_lr_max,
                    perf_bkg.z_pred_max - perf_bkg.z_lr_max,
                    z_true_max_latex,
                    delta_zmax_latex,
                    ref=0,
                    xlim=(0, max(perf_bkg.z_lr_max)),
                    ylim=config['conditions'].get('deltazmax_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_vs_ZLRmax_background.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_bkg.get_mean_std_hist(
                        perf_bkg.z_lr_max,
                        perf_bkg.z_pred_max - perf_bkg.z_lr_max,
                        z_true_max_latex,
                        delta_zmax_latex,
                        xlim=(0, max(perf_bkg.z_lr_max)),
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZmax_vs_ZLRmax_background_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZmax_vs_ZLRmax_background_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

        if ('deltazmax_vs_zLRmax_bin_ratio' in what_to_plot or 'all' in what_to_plot):
            if perf_sig and perf_sig.z_lr is not None:
                fig = perf_sig.get_2d_hist(
                    perf_sig.z_lr_max_bin/perf_sig.n_bins,
                    perf_sig.z_pred_max - perf_sig.z_lr_max,
                    z_true_max_latex + ' bin / number of bins',
                    delta_zmax_latex,
                    ref=0,
                    interpolate=config['conditions'].get('interpolate',None),
                    ylim=config['conditions'].get('deltazmax_lim',None),
                    xlim=(0,1),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_vs_bin_ratio_signal.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_sig.get_mean_std_hist(
                        perf_sig.z_lr_max_bin/perf_sig.n_bins,
                        perf_sig.z_pred_max - perf_sig.z_lr_max,
                        z_true_max_latex + ' bin / number of bins',
                        delta_zmax_latex,
                        xlim=(0, 1),
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZmax_vs_bin_ratio_signal_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZmax_vs_bin_ratio_signal_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

            if perf_bkg and perf_bkg.z_lr is not None:
                fig = perf_bkg.get_2d_hist(
                    perf_bkg.z_lr_max_bin/perf_bkg.n_bins,
                    perf_bkg.z_pred_max - perf_bkg.z_lr_max,
                    z_true_max_latex + ' bin  / number of bins',
                    delta_zmax_latex,
                    ref=0,
                    interpolate=config['conditions'].get('interpolate',None),
                    ylim=config['conditions'].get('deltazmax_lim',None),
                    xlim=(0,1),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_vs_bin_ratio_background.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_bkg.get_mean_std_hist(
                        perf_bkg.z_lr_max_bin/perf_bkg.n_bins,
                        perf_bkg.z_pred_max - perf_bkg.z_lr_max,
                        z_true_max_latex + ' bin / number of bins',
                        delta_zmax_latex,
                        xlim=(0, 1),
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZmax_vs_bin_ratio_background_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZmax_vs_bin_ratio_background_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

        if ('deltazmax_vs_zLRmax_bin_number' in what_to_plot or 'all' in what_to_plot):
            if perf_sig and perf_sig.z_lr is not None:
                fig = perf_sig.get_2d_hist(
                    perf_sig.z_lr_max_bin,
                    perf_sig.z_pred_max - perf_sig.z_lr_max,
                    z_true_max_latex + ' bin',
                    delta_zmax_latex,
                    ref=0,
                    interpolate=config['conditions'].get('interpolate',None),
                    ylim=config['conditions'].get('deltazmax_lim',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_vs_bin_number_signal.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_sig.get_mean_std_hist(
                        perf_sig.z_lr_max_bin,
                        perf_sig.z_pred_max - perf_sig.z_lr_max,
                        z_true_max_latex + ' bin',
                        delta_zmax_latex,
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZmax_vs_bin_number_signal_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZmax_vs_bin_number_signal_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

            if perf_bkg and perf_bkg.z_lr is not None:
                fig = perf_bkg.get_2d_hist(
                    perf_bkg.z_lr_max_bin,
                    perf_bkg.z_pred_max - perf_bkg.z_lr_max,
                    z_true_max_latex + ' bin',
                    delta_zmax_latex,
                    ref=0,
                    interpolate=config['conditions'].get('interpolate',None),
                    ylim=config['conditions'].get('deltazmax_lim',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_vs_bin_number_background.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_bkg.get_mean_std_hist(
                        perf_bkg.z_lr_max_bin,
                        perf_bkg.z_pred_max - perf_bkg.z_lr_max,
                        z_true_max_latex + ' bin',
                        delta_zmax_latex,
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZmax_vs_bin_number_background_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZmax_vs_bin_number_background_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

        if ('deltazmax_vs_entries_first_bin' in what_to_plot or 'all' in what_to_plot):
            if perf_sig:
                fig = perf_sig.get_2d_hist(
                    perf_sig.entries_first_bin,
                    perf_sig.z_pred_max - perf_sig.z_lr_max,
                    r'Entries in first bin',
                    delta_zmax_latex,
                    ref=0,
                    ylim=config['conditions'].get('deltazmax_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_vs_entries_first_bin_signal.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_sig.get_mean_std_hist(
                        perf_sig.entries_first_bin,
                        perf_sig.z_pred_max - perf_sig.z_lr_max,
                        r'Entries in first bin',
                        delta_zmax_latex,
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZmax_vs_entries_first_bin_signal_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZmax_vs_entries_first_bin_signal_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

            if perf_bkg and perf_bkg.z_lr is not None:
                fig = perf_bkg.get_2d_hist(
                    perf_bkg.entries_first_bin,
                    perf_bkg.z_pred_max - perf_bkg.z_lr_max,
                    r'Entries in first bin',
                    delta_zmax_latex,
                    ref=0,
                    ylim=config['conditions'].get('deltazmax_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_vs_entries_first_bin_background.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_bkg.get_mean_std_hist(
                        perf_bkg.entries_first_bin,
                        perf_bkg.z_pred_max - perf_bkg.z_lr_max,
                        r'Entries in first bin',
                        delta_zmax_latex,
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZmax_vs_entries_first_bin_background_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZmax_vs_entries_first_bin_background_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

        if ('deltazmax_vs_entries_last_bin' in what_to_plot or 'all' in what_to_plot):
            if perf_sig:
                fig = perf_sig.get_2d_hist(
                    perf_sig.entries_last_bin,
                    perf_sig.z_pred_max - perf_sig.z_lr_max,
                    r'Entries in last bin',
                    delta_zmax_latex,
                    ref=0,
                    ylim=config['conditions'].get('deltazmax_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_vs_entries_last_bin_signal.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_sig.get_mean_std_hist(
                        perf_sig.entries_last_bin,
                        perf_sig.z_pred_max - perf_sig.z_lr_max,
                        r'Entries in last bin',
                        delta_zmax_latex,
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZmax_vs_entries_last_bin_signal_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZmax_vs_entries_last_bin_signal_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

            if perf_bkg and perf_bkg.z_lr is not None:
                fig = perf_bkg.get_2d_hist(
                    perf_bkg.entries_last_bin,
                    perf_bkg.z_pred_max - perf_bkg.z_lr_max,
                    r'Entries in last bin',
                    delta_zmax_latex,
                    ref=0,
                    ylim=config['conditions'].get('deltazmax_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_vs_entries_last_bin_background.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_bkg.get_mean_std_hist(
                        perf_bkg.entries_last_bin,
                        perf_bkg.z_pred_max - perf_bkg.z_lr_max,
                        r'Entries in last bin',
                        delta_zmax_latex,
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZmax_vs_entries_last_bin_background_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZmax_vs_entries_last_bin_background_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

        if ('deltazmax_vs_B_entries_sig_bin' in what_to_plot or 'all' in what_to_plot):
            if perf_sig:
                fig = perf_sig.get_2d_hist(
                    perf_sig.B_entries_sig_bin,
                    (perf_sig.z_pred_max - perf_sig.z_lr_max),
                    r'Background in signal bin',
                    delta_zmax_latex,
                    ref=0,
                    ylim=config['conditions'].get('deltazmax_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    include_mean=False,
                    include_sd=False,
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_vs_B_sig_bin.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_sig.get_mean_std_hist(
                        perf_sig.B_entries_sig_bin,
                        perf_sig.z_pred_max - perf_sig.z_lr_max,
                        r'Background in signal bin',
                        delta_zmax_latex,
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZmax_vs_B_sig_bin_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZmax_vs_B_sig_bin_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

        if ('deltazmax_vs_entries_zLRmax_bin' in what_to_plot or 'all' in what_to_plot):
            if perf_sig:
                fig = perf_sig.get_2d_hist(
                    perf_sig.entries_lr,
                    perf_sig.z_pred_max - perf_sig.z_lr_max,
                    'Entries ' + z_true_max_latex + 'bin',
                    delta_zmax_latex,
                    ref=0,
                    ylim=config['conditions'].get('deltazmax_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_vs_entries_ZLRmax_bin_signal.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_sig.get_mean_std_hist(
                        perf_sig.entries_lr,
                        perf_sig.z_pred_max - perf_sig.z_lr_max,
                        'Entries ' + z_true_max_latex + 'bin',
                        delta_zmax_latex,
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZmax_vs_entries_ZLRmax_bin_signal_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZmax_vs_entries_ZLRmax_bin_signal_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

            if perf_bkg and perf_bkg.z_lr is not None:
                fig = perf_bkg.get_2d_hist(
                    perf_bkg.entries_lr,
                    perf_bkg.z_pred_max - perf_bkg.z_lr_max,
                    'Entries ' + z_true_max_latex + 'bin',
                    delta_zmax_latex,
                    ref=0,
                    ylim=config['conditions'].get('deltazmax_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_vs_entries_ZLRmax_bin_background.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_bkg.get_mean_std_hist(
                        perf_bkg.entries_lr,
                        perf_bkg.z_pred_max - perf_bkg.z_lr_max,
                        'Entries ' + z_true_max_latex + 'bin',
                        delta_zmax_latex,
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZmax_vs_entries_ZLRmax_bin_background_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZmax_vs_entries_ZLRmax_bin_background_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

        if ('deltazmax_vs_entries_sig_bin' in what_to_plot or 'all' in what_to_plot):
            if perf_sig:
                fig = perf_sig.get_2d_hist(
                    perf_sig.entries_sig_bin,
                    perf_sig.z_pred_max - perf_sig.z_lr_max,
                    r'Entries signal bin',
                    delta_zmax_latex,
                    ref=0,
                    ylim=config['conditions'].get('deltazmax_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_vs_entries_sig_bin_signal.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_sig.get_mean_std_hist(
                        perf_sig.entries_sig_bin,
                        perf_sig.z_pred_max - perf_sig.z_lr_max,
                        r'Entries signal bin',
                        delta_zmax_latex,
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZmax_vs_entries_sig_bin_signal_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZmax_vs_entries_sig_bin_signal_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

        if ('deltazmax_vs_B_zLRmax_bin' in what_to_plot or 'all' in what_to_plot):
            if perf_sig:
                fig = perf_sig.get_2d_hist(
                    perf_sig.bkg_lr,
                    perf_sig.z_pred_max - perf_sig.z_lr_max,
                    'Background at ' + z_true_max_latex + 'bin',
                    delta_zmax_latex,
                    ref=0,
                    ylim=config['conditions'].get('deltazmax_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_vs_bkg_zLRmax_bin_signal.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_sig.get_mean_std_hist(
                        perf_sig.bkg_lr,
                        perf_sig.z_pred_max - perf_sig.z_lr_max,
                        'Background at ' + z_true_max_latex + 'bin',
                        delta_zmax_latex,
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZmax_vs_bkg_zLRmax_bin_signal_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZmax_vs_bkg_zLRmax_bin_signal_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

            if perf_bkg and perf_bkg.z_lr is not None:
                fig = perf_bkg.get_2d_hist(
                    perf_bkg.bkg_lr,
                    perf_bkg.z_pred_max - perf_bkg.z_lr_max,
                    'Background at ' + z_true_max_latex + 'bin',
                    delta_zmax_latex,
                    ref=0,
                    ylim=config['conditions'].get('deltazmax_lim',None),
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_vs_bkg_zLRmax_bin_background.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_bkg.get_mean_std_hist(
                        perf_bkg.bkg_lr,
                        perf_bkg.z_pred_max - perf_bkg.z_lr_max,
                        'Background at ' + z_true_max_latex + 'bin',
                        delta_zmax_latex,
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZmax_vs_bkg_zLRmax_bin_background_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZmax_vs_bkg_zLRmax_bin_background_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')


        if ('deltazmax_vs_bins' in what_to_plot or 'all' in what_to_plot):
            if perf_sig:
                fig = perf_sig.get_2d_hist(
                    perf_sig.n_bins,
                    perf_sig.z_pred_max - perf_sig.z_lr_max,
                    r'Number of bins',
                    delta_zmax_latex,
                    ref=0,
                    xlim=(10,110),
                    interpolate=config['conditions'].get('interpolate',None),
                    ylim=config['conditions'].get('deltazmax_lim',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_vs_bins_signal.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_sig.get_mean_std_hist(
                        perf_sig.n_bins,
                        perf_sig.z_pred_max - perf_sig.z_lr_max,
                        r'Number of bins',
                        delta_zmax_latex,
                        xlim=(10,110),
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZmax_vs_bins_signal_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZmax_vs_bins_signal_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

            if perf_bkg and perf_bkg.z_lr is not None:
                fig = perf_bkg.get_2d_hist(
                    perf_bkg.n_bins,
                    perf_bkg.z_pred_max - perf_bkg.z_lr_max,
                    r'Number of bins',
                    delta_zmax_latex,
                    ref=0,
                    xlim=(10,110),
                    interpolate=config['conditions'].get('interpolate',None),
                    ylim=config['conditions'].get('deltazmax_lim',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_vs_bins_background.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

                if config['plot_std_dist']:
                    fig_mean, fig_std = perf_bkg.get_mean_std_hist(
                        perf_bkg.n_bins,
                        perf_bkg.z_pred_max - perf_bkg.z_lr_max,
                        r'Number of bins',
                        delta_zmax_latex,
                        xlim=(10,110),
                        ylim=config['conditions'].get('deltaz_lim',None),
                        show=config.get('show',False)
                    )
                    save_path_mean = output_dir/f'DeltaZmax_vs_bins_background_mean.{fmt}'
                    fig_mean.savefig(str(save_path_mean), bbox_inches='tight')
                    plt.close(fig_mean)
                    print(f'Saved {save_path_mean}')

                    save_path_std = output_dir/f'DeltaZmax_vs_bins_background_std.{fmt}'
                    fig_std.savefig(str(save_path_std), bbox_inches='tight')
                    plt.close(fig_std)
                    print(f'Saved {save_path_std}')

        if ('deltazmax_bin_vs_zLRmax_bin_number' in what_to_plot or 'all' in what_to_plot):
            if perf_sig:
                fig = perf_sig.get_2d_hist(
                    perf_sig.z_lr_max_bin,
                    perf_sig.z_pred_max_bin - perf_sig.z_lr_max_bin,
                    z_true_max_latex + ' bin',
                    delta_zmax_bin_latex,
                    ref=0,
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_bin_vs_zLRmax_bin_number_signal.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

            if perf_bkg and perf_bkg.z_lr is not None:
                fig = perf_bkg.get_2d_hist(
                    perf_bkg.z_lr_max_bin,
                    perf_bkg.z_pred_max_bin - perf_bkg.z_lr_max_bin,
                    z_true_max_latex + ' bin',
                    delta_zmax_bin_latex,
                    ref=0,
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_bin_vs_zLRmax_bin_number_background.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

        if ('deltazmax_bin_vs_zLRmax_bin_ratio' in what_to_plot or 'all' in what_to_plot):
            if perf_sig:
                fig = perf_sig.get_2d_hist(
                    perf_sig.z_lr_max_bin/perf_sig.n_bins,
                    (perf_sig.z_pred_max_bin - perf_sig.z_lr_max_bin)/perf_sig.n_bins,
                    z_true_max_latex + ' bin / Number of bins',
                    delta_zmax_bin_ratio_latex,
                    ref=0,
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_bin_vs_zLRmax_bin_ratio_signal.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

            if perf_bkg and perf_bkg.z_lr is not None:
                fig = perf_bkg.get_2d_hist(
                    perf_bkg.z_lr_max_bin/perf_bkg.n_bins,
                    (perf_bkg.z_pred_max_bin - perf_bkg.z_lr_max_bin)/perf_bkg.n_bins,
                    z_true_max_latex + ' bin / Number of bins',
                    delta_zmax_bin_ratio_latex,
                    ref=0,
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_bin_vs_zLRmax_bin_ratio_background.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

        if ('deltazmax_bin_vs_zLRmax_number' in what_to_plot or 'all' in what_to_plot):
            if perf_sig:
                fig = perf_sig.get_2d_hist(
                    perf_sig.z_lr_max,
                    perf_sig.z_pred_max_bin - perf_sig.z_lr_max_bin,
                    z_true_max_latex,
                    delta_zmax_bin_latex,
                    ref=0,
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZ_bin_vs_zLRmax_number_signal.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

            if perf_bkg and perf_bkg.z_lr is not None:
                fig = perf_bkg.get_2d_hist(
                    perf_bkg.z_lr_max,
                    perf_bkg.z_pred_max_bin - perf_bkg.z_lr_max_bin,
                    z_true_max_latex,
                    delta_zmax_bin_latex,
                    ref=0,
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_bin_vs_zLRmax_number_background.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')
                
        if ('deltazmax_bin_vs_zLRmax_ratio' in what_to_plot or 'all' in what_to_plot):
            if perf_sig:
                fig = perf_sig.get_2d_hist(
                    perf_sig.z_lr_max,
                    (perf_sig.z_pred_max_bin - perf_sig.z_lr_max_bin)/perf_sig.n_bins,
                    z_true_max_latex,
                    delta_zmax_bin_ratio_latex,
                    ref=0,
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_bin_vs_zLRmax_ratio_signal.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

            if perf_bkg and perf_bkg.z_lr is not None:
                fig = perf_bkg.get_2d_hist(
                    perf_bkg.z_lr_max,
                    (perf_bkg.z_pred_max_bin - perf_bkg.z_lr_max_bin)/perf_bkg.n_bins,
                    z_true_max_latex,
                    delta_zmax_bin_ratio_latex,
                    ref=0,
                    interpolate=config['conditions'].get('interpolate',None),
                    show=config.get('show',False)
                )
                save_path = output_dir/f'DeltaZmax_bin_vs_zLRmax_ratio_background.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')
	
	

        if ('deltaBpred_distributions' in what_to_plot) and perf_sig and perf_bkg:
            B = np.concatenate(perf_sig.background)
            Bpred = np.concatenate(perf_sig.bpred)
            SQRT = np.clip(np.sqrt(B), 1, None)
            hist1, x1 = pth.get_1d_hist((B - Bpred)/SQRT,
                                        binning=np.linspace(-5, 5, 100))
            mean1 = np.sum(hist1 * x1) / np.sum(hist1)
            std1 = np.sum(hist1 * (x1 - mean1)**2) / np.sum(hist1)

            B = np.concatenate(perf_bkg.background)
            Bpred = np.concatenate(perf_bkg.bpred)
            SQRT = np.clip(np.sqrt(B), 1, None)
            hist2, x2 = pth.get_1d_hist((B - Bpred)/SQRT,
                                        binning=np.linspace(-5, 5, 100))
            mean2 = np.sum(hist2 * x2) / np.sum(hist2)
            std2 = np.sum(hist2 * (x2 - mean2)**2) / np.sum(hist2)
            

            fig, ax = plt.subplots(1, 1, figsize=(10/2.54, 10/2.54), dpi=dpi)
            ax.errorbar(x1, hist1,
                        label=f'S+B\n($\mu = ${mean1:.2f}, $\sigma = ${std1:.2g})',
                        drawstyle='steps-mid', color='tab:orange', linestyle='-', linewidth=0.7)
            ax.errorbar(x2, hist2,
                        label=f'B\n($\mu = ${mean2:.2f}, $\sigma = ${std2:.2g})',
                        drawstyle='steps-mid', color='tab:blue', linestyle='-', linewidth=0.7)
            ax.axvline(x=0, linestyle='--', linewidth=0.7, color='black', alpha=0.3)
            ymax = max(max(hist1),max(hist2))
            ax.set_ylim(bottom=0.5, top=1.25*ymax)
            ax.legend(loc='upper right', frameon=False)
            ax.set_xlabel(delta_bkg_latex, loc='right')
            ax.set_ylabel('Entries', loc='top')
            ax.set_xlim(-5, +5)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
            ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)
            ax.tick_params(which='major', axis='both')
            ax.tick_params(which='minor', axis='both')
            if config.get('show', False): plt.show()
            save_path = output_dir/f'DeltaB.{fmt}'
            fig.savefig(str(save_path), bbox_inches='tight')
            plt.yscale('log')
            save_path = output_dir/f'DeltaB_log.{fmt}'
            fig.savefig(str(save_path), bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {save_path}')

        if ('false_discovery_rate_plot' in what_to_plot or 'all' in what_to_plot):

            # Plot discovery rates
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)
            data = {'BumpNet':(perf_bkg.z_pred_max,'tab:blue')}
            if perf_bkg.z_lr_max is not None:
                data['LR'] = (perf_bkg.z_lr_max, 'tab:green')
            for label, (z, color) in data.items():
                x = np.arange(3, 7.5, 0.5)
                y  = np.array([z[np.where(z >= thr)[0]].shape[0]/z.shape[0]*100 for thr in x])
                ax.errorbar(x, y, label=label, color=color, marker='.', linewidth=0.7, linestyle='-')
                ref = y[np.where(x==5)[0][0]]
                ax.axhline(ref, xmax=(5-min(x))/(max(x)-min(x)), linestyle='--', linewidth=0.7, color=color)
                ax.annotate(f'{ref:.2g}%', xy=(min(x),ref), xycoords='data', fontsize=8, color=color)

            ax.axvline(x=5, linestyle='--', linewidth=0.7, color='black', alpha=0.1)
            ax.legend(loc='upper right', frameon=False)
            ax.set_xlabel(z_max_latex, loc='right')
            ax.set_ylabel('False discovery rate (%)', loc='top')
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
            ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)
            ax.tick_params(which='major', axis='both')
            ax.tick_params(which='minor', axis='both')
            if config.get('show', False): plt.show()
            save_path = output_dir/f'false_discovery_rate_plot.{fmt}'
            fig.savefig(str(save_path), bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {save_path}')

        if ('false_discovery_rate_table' in what_to_plot or 'all' in what_to_plot):

            fig, ax = plt.subplots(1, 1, figsize=(11/2.54, 11/2.54), dpi=dpi)
            fig.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')
            z_values = {1:('BumpNet', perf_bkg.z_pred_max)}
            columns = ['', '', 'BumpNet', '']
            if perf_bkg.z_lr_max is not None:
                z_values[2] = ('LR', perf_bkg.z_lr_max)
                columns += ['LR']
            rows = np.arange(3, 8, 1)
            ax.set_ylim(-1, len(rows) + 1)
            ax.set_xlim(0, len(columns) + 0.5)

            # Add rows text
            for j, column in enumerate(columns):
                if j in [1,3]: continue #skip empty columns
                for i, thr in enumerate(reversed(rows)):
                    if j == 0:
                        text = z_max_latex+f'$\\geq {thr}$'
                    elif j==2 or j==4:
                        z = z_values[j/2][1]
                        discovery = z[np.where(z >= thr)[0]]
                        text = f'{(discovery.shape[0]/z.shape[0])*100:.3g}% ({discovery.shape[0]})'
                    ax.annotate(
                        xy=(j + 0.5, i + 0.5),
                        text=text,
                        ha='center',
                        va='center',
                        weight='normal'
                    )

            # Add column names
            for j, column in enumerate(columns):
                ax.annotate(
                    xy=(j + 0.5, len(rows) + .5),
                    text=column,
                    ha='center',
                    va='center',
                    weight='bold'
                )

            # Add dividing lines
            ax.plot([0, len(columns) + 1], [len(rows)+1, len(rows)+1], lw='.5', c='black')
            ax.plot([0, len(columns) + 1], [len(rows), len(rows)], lw='.5', c='black')
            ax.plot([0, len(columns) + 1], [0, 0], lw='.5', c='black')
            for i, row in enumerate(rows):
                ax.plot(
                    [0, len(columns) + 1],
                    [i, i],
                    ls=':',
                    lw='.5',
                    c='grey'
                )
            ax.set_title('False discovery rates')
            if config.get('show', False): plt.show()
            save_path = output_dir/f'false_discovery_rate_table.{fmt}'
            fig.savefig(str(save_path), bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {save_path}')

        if ('outlier_rate_plot' in what_to_plot or 'all' in what_to_plot):

            # Plot discovery rates
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)
            zmax_thr = config['conditions'].get('outliers_zmax_threshold', 5)
            dz_values = {}
            if perf_bkg:
                z = perf_bkg.z_pred_max
                dz = perf_bkg.z_pred_max - perf_bkg.z_lr_max
                label = 'B'
                std = 1
                if config['conditions'].get('outliers_in_std_units', False):
                    std = np.std(dz)
                    label += f' ($\\sigma$ = {std:.2g})'
                dz_values[label] = (z, dz, std, 'tab:blue')
            if perf_sig:
                z = perf_sig.z_pred_max
                dz = perf_sig.z_pred_max - perf_sig.z_lr_max
                label = 'S+B'
                std = 1
                if config['conditions'].get('outliers_in_std_units', False):
                    std = np.std(dz)
                    label += f' ($\\sigma$ = {std:.2g})'
                dz_values[label] = (z, dz, std, 'tab:orange')
            for label, (z, dz, std, color) in dz_values.items():
                x = np.arange(-6, 7, 1)
                x = x[np.where(x!=0)[0]]
                def get_rate(thr, dz):
                    discovery = np.where(z >= zmax_thr)[0]
                    if thr < 0:
                        outlier = np.where(np.abs(dz) <= thr)[0]
                    else:
                        outlier = np.where(np.abs(dz) >= thr)[0]
                    both = np.intersect1d(discovery, outlier)
                    return (both.shape[0]/z.shape[0])*100

                y  = np.array([get_rate(thr*std, dz) for thr in x])
                ax.errorbar(x, y, label=label, color=color, marker='.', linewidth=0.7, linestyle='-')

            ax.axvline(x=0, linestyle='--', linewidth=0.7, color='black', alpha=0.1)
            ax.legend(loc='upper right', frameon=False)
            xlabel = delta_zmax_latex
            if config['conditions'].get('outliers_in_std_units', False):
                xlabel += '$\\sigma$'
            ax.set_title(z_max_latex + '$ \\geq '+f'{zmax_thr}$')
            ax.set_xlabel(xlabel, loc='right')
            ax.set_ylabel('Outlier rate (%)', loc='top')
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
            ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)
            ax.tick_params(which='major', axis='both')
            ax.tick_params(which='minor', axis='both')
            if config.get('show', False): plt.show()
            save_path = output_dir/f'outlier_rate_plot.{fmt}'
            fig.savefig(str(save_path), bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {save_path}')

        if ('outlier_deltazmax_plot' in what_to_plot or 'all' in what_to_plot) and perf_sig and perf_bkg:

            zmax_thr = config['conditions'].get('outliers_zmax_threshold', None)
            dz = perf_sig.z_pred_max - perf_sig.z_lr_max
            if zmax_thr is not None:
                idx = np.where(perf_sig.z_pred_max >= zmax_thr)[0]
                dz = dz[idx]
            hist1, x1 = pth.get_1d_hist(dz)
            mean1 = np.sum(hist1 * x1) / np.sum(hist1)
            std1 = np.sum(hist1 * (x1 - mean1)**2) / np.sum(hist1)

            dz = perf_bkg.z_pred_max - perf_bkg.z_lr_max
            if zmax_thr is not None:
                idx = np.where(perf_bkg.z_pred_max >= zmax_thr)[0]
                dz = dz[idx]
            hist2, x2 = pth.get_1d_hist(dz)
            mean2 = np.sum(hist2 * x2) / np.sum(hist2)
            std2 = np.sum(hist2 * (x2 - mean2)**2) / np.sum(hist2)

            # Plot
            gridspec_kw = {'height_ratios': [1,1], 'hspace': 0.05}
            fig, ax = plt.subplots(2, 1, figsize=(10/2.54, 10/2.54), dpi=dpi,
                                   sharex=True, gridspec_kw=gridspec_kw)


            for k, (label, x, hist, mu, sigma, color) in enumerate([
                    (f'S+B ($\mu = ${mean1:.2f}, $\sigma = ${std1:.2g})', x1, hist1, mean1, std1, 'tab:orange'),
                    (f'B ($\mu = ${mean2:.2f}, $\sigma = ${std2:.2g})', x2, hist2, mean2, std2, 'tab:blue')
            ]):
                xdata, ydata = x, hist
                xlabel = delta_zmax_latex
                deltazmax_thr = config['conditions'].get('outliers_deltazmax_threshold', 5)
                xleft, xright = (-1*deltazmax_thr, deltazmax_thr)
                xrange = r'$|\Delta Z_{\mathrm{max}}| \leq '+f'{deltazmax_thr}$'

                if config['conditions'].get('outliers_in_std_units', True):
                    xleft = xleft*sigma
                    xright = xright*sigma
                    xrange += '$\\times\\sigma$'

                # Plot deltazmax distribution
                ax[k].errorbar(xdata, ydata,
                               label=label, color=color, drawstyle='steps-mid', linestyle='-', linewidth=0.7)


                # Plot gaussian fit with same mu/sigma as deltazmax
                from scipy.optimize import curve_fit
                def Gauss(x, A):
                    y = A * 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )
                    return y
                parameters, covariance = curve_fit(Gauss, xdata, ydata)
                fit = Gauss(xdata, *parameters)
                ax[k].errorbar(xdata, fit, color=color,
                            label='Gaussian', #'Gaussian fit ($\\mu$ = '+f'{parameters[0]:.2}'+', $\\sigma$ = '+f'{parameters[1]:.2}'+')',
                            linestyle='--', linewidth=0.7)

                ax[k].axvline(x=0, linestyle='-', linewidth=0.7, color='black', alpha=0.1)
                idx = np.where(xdata <= xright)[0]
                ax[k].fill_between(xdata[idx], 0, ydata[idx], step='mid', color=color, alpha=0.1,
                                   label=xrange)
                idx = np.where(xdata >= xleft)[0]
                ax[k].fill_between(xdata[idx], 0, ydata[idx], step='mid', color=color, alpha=0.1,)
                idx = np.where(xdata <= xleft)[0]
                ax[k].fill_between(xdata[idx], 0, ydata[idx], step='mid', color='tab:red', alpha=0.5,
                                   label = 'Outliers')
                idx = np.where(xdata >= xright)[0]
                ax[k].fill_between(xdata[idx], 0, ydata[idx], step='mid', color='tab:red', alpha=0.5)

                ax[k].set_yscale('log')
                ax[k].set_ylim(1e-1, 1e5)
                ax[k].legend(loc='upper left', ncols=2, frameon=False, fontsize=8)
                if k == 1: ax[k].set_xlabel(xlabel, loc='right')
                ax[k].set_ylabel('Entries', loc='top')
                ax[k].tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
                ax[k].tick_params(axis='y', which='both', direction='in', left=True, right=True)
                ax[k].tick_params(which='major', axis='both')
                ax[k].tick_params(which='minor', axis='both')
            if config.get('show', False): plt.show()
            save_path = output_dir/f'outlier_DeltaZmax.{fmt}'
            fig.savefig(str(save_path), bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {save_path}')

        if ('outlier_rate_table' in what_to_plot or 'all' in what_to_plot):

            fig, ax = plt.subplots(1, 1, figsize=(10/2.54, 10/2.54), dpi=dpi)
            fig.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')
            dz_values = {}
            columns = ['','','']
            zmax_thr = config['conditions'].get('outliers_zmax_threshold', None)

            if perf_bkg:
                z = perf_bkg.z_pred_max
                total = z.shape[0]
                dz = perf_bkg.z_pred_max - perf_bkg.z_lr_max
                label = 'B'
                std = 1
                if config['conditions'].get('outliers_in_std_units', False):
                    std = np.std(dz)
                    label += '\n'+f'($\\sigma$ = {std:.2g})'
                if zmax_thr:
                    idx = np.where(z >= zmax_thr)[0]
                    dz = dz[idx]
                dz_values[1] = (label, z, dz, std, total)
                columns += [label , '']
            if perf_sig:
                z = perf_sig.z_pred_max
                total = z.shape[0]
                dz = perf_sig.z_pred_max - perf_sig.z_lr_max
                label = 'S+B'
                std = 1
                if config['conditions'].get('outliers_in_std_units', False):
                    std = np.std(dz)
                    label += '\n'+f'($\\sigma$ = {std:.2g})'
                if zmax_thr:
                    idx = np.where(z >= zmax_thr)[0]
                    dz = dz[idx]
                dz_values[2] = (label, z, dz, std, total)
                columns += [label , '']
            rows = np.arange(1, 6, 1)
            ax.set_ylim(-1, len(rows) + 1)
            ax.set_xlim(0, len(columns) + 0.5)


            # Add rows text
            for j, column in enumerate(columns):
                if j in [0,2,4,6]: continue #skip empty columns
                for i, thr in enumerate(reversed(rows)):
                    if j == 1:
                        text = '$|\\Delta Z_{\\mathrm{max}}|$'+'$\\geq$ '+f'{thr}'
                        if config['conditions'].get('outliers_in_std_units', False):
                            text += r'$\times\sigma$'
                    else:
                        z = dz_values[(j-1)/2][1]
                        dz = dz_values[(j-1)/2][2]
                        std = dz_values[(j-1)/2][3]
                        total = dz_values[(j-1)/2][4]
                        outlier = np.where(np.abs(dz) >= thr*std)[0]
                        text = f'{(outlier.shape[0]/total)*100:.3g}% ({outlier.shape[0]})'
                    ax.annotate(
                        xy=(j + 0.5, i + 0.5),
                        text=text,
                        ha='center',
                        va='center',
                        weight='normal',
                        fontsize=8
                    )

            # Add column names
            for j, column in enumerate(columns):
                ax.annotate(
                    xy=(j + 0.5, len(rows) + .5),
                    text=column,
                    ha='center',
                    va='center',
                    weight='bold'
                )

            # Add dividing lines
            ax.plot([0, len(columns) + 1], [len(rows)+1, len(rows)+1], lw='.5', c='black')
            ax.plot([0, len(columns) + 1], [len(rows), len(rows)], lw='.5', c='black')
            ax.plot([0, len(columns) + 1], [0, 0], lw='.5', c='black')
            for i, row in enumerate(rows):
                ax.plot(
                    [0, len(columns) + 1],
                    [i, i],
                    ls=':',
                    lw='.5',
                    c='grey'
                )
            ax.set_title('Outlier rates ('+ z_max_latex + '$ \\geq '+f'{zmax_thr}$)')
            if config.get('show', False): plt.show()
            save_path = output_dir/f'outlier_rate_table.{fmt}'
            fig.savefig(str(save_path), bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {save_path}')

        if ('predictions_heatmap' in what_to_plot or 'all' in what_to_plot):
            if perf_sig is not None:
                fig = perf_sig.get_predictions_heatmap(
                    mass_binning=eval(config['conditions'].get('predictions_mass_binning', 'None'))
                )
                if config.get('show', False): plt.show()
                save_path = output_dir/f'predictions_heatmap_signal.{fmt}'
                fig.savefig(save_path, bbox_inches='tight')
                print(f'Saved {save_path}')
                plt.close(fig)

            if perf_bkg is not None:
                fig = perf_bkg.get_predictions_heatmap(
                    mass_binning=eval(config['conditions'].get('predictions_mass_binning','None'))
                )
                if config.get('show', False): plt.show()
                save_path = output_dir/f'predictions_heatmap_background.{fmt}'
                fig.savefig(save_path, bbox_inches='tight')
                print(f'Saved {save_path}')
                plt.close(fig)

        if 'z_pred_corrCoeff' in what_to_plot or 'all' in what_to_plot:


            if perf_sig is not None or perf_bkg is not None:
                fig, ax = plt.subplots(1, 1, figsize=(10/2.54, 10/2.54), dpi=dpi)
                thr = 0.9
                if perf_sig is not None:
                    frac_sig = (perf_sig.z_pred_corrCoeff > thr).sum() / perf_sig.z_pred_corrCoeff.shape[0]
                    ax.hist(perf_sig.z_pred_corrCoeff, bins=np.linspace(0.3, 1, 35), density=True, histtype='stepfilled',
                                alpha=0.3, color='tab:orange',
                                label=f'fraction above 0.9, S+B: {frac_sig:.3g}')
                    # compute the fraction of events with z_pred_corrCoeff > 0.9
                    
                if perf_bkg is not None:
                    frac_bkg = (perf_bkg.z_pred_corrCoeff > thr).sum() / perf_bkg.z_pred_corrCoeff.shape[0]
                    ax.hist(perf_bkg.z_pred_corrCoeff, bins=np.linspace(0.3, 1, 35), density=True, histtype='stepfilled',
                                alpha=0.3, color='tab:blue',
                                label=f'fraction above 0.9, B: {frac_bkg:.3g}')
                ax.legend(loc='upper left', frameon=False)
                ax.set_xlabel(z_pred_corrCeff_latex, loc='right')
                ax.set_ylabel('Density', loc='top')
                save_path = output_dir/f'z_pred_corrCeff.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

        if ('examples' in what_to_plot or 'all' in what_to_plot):

            counter=0
            if perf_sig:
                for i in range(perf_sig.n_hists):
                    condition = conditions.get('signal_examples_condition', 'True')
                    selection = conditions.get('selection', {})
                    hist_name = perf_sig.names[i][0]
                    if not eval(condition) or not passes_selection(hist_name, selection): continue
                    if counter >= conditions['n_examples']: break
                    fig = perf_sig.get_example(i, show=config.get('show',False))
                    zlr_max_sig = f'zlrmax={perf_sig.z_lr_max[i]:.2f}_' if conditions.get('order_examples_by_significance', False) else ''
                    save_path = output_dir/f'{zlr_max_sig}example_signal_{perf_sig.names[i][0]}.{fmt}'
                    fig.savefig(str(save_path), bbox_inches='tight')
                    plt.close(fig)
                    print(f'Saved {save_path}')
                    counter+=1

            counter=0
            if perf_bkg:
                for i in range(perf_bkg.n_hists):
                    condition = conditions.get('background_examples_condition', 'True')
                    selection = conditions.get('selection', {})
                    hist_name = perf_bkg.names[i][0]
                    if not eval(condition) or not passes_selection(hist_name, selection): continue
                    if counter >= conditions['n_examples']: break
                    fig = perf_bkg.get_example(i, show=config.get('show',False))
                    zlr_max_bkg = f'zlrmax={perf_bkg.z_lr_max[i]:.2f}_' if conditions.get('order_examples_by_significance', False) else ''
                    save_path = output_dir/f'{zlr_max_bkg}example_background_{perf_bkg.names[i][0]}.{fmt}'
                    fig.savefig(str(save_path), bbox_inches='tight')
                    plt.close(fig)
                    print(f'Saved {save_path}')
                    counter+=1
        
        if 'systematic_ratio' in what_to_plot and 'NOSYS' not in dataset_config['bkg_input_dir']:
            if perf_nominal:
                counter=0
                if perf_sig:
                    for i in range(perf_sig.n_hists):
                        condition = conditions.get('signal_examples_condition', 'True')
                        if not eval(condition): continue
                        if counter >= conditions['n_examples']: break
                        fig = perf_sig.get_systematic_ratio(i, perf_nominal, show=config.get('show',False))
                        save_path = output_dir/f'systematic_ratio_signal_{perf_sig.names[i][0]}.{fmt}'
                        if fig is None:
                            continue
                        fig.savefig(str(save_path), bbox_inches='tight')
                        plt.close(fig)
                        print(f'Saved {save_path}')
                        counter+=1

                counter=0
                if perf_bkg:
                    for i in range(perf_bkg.n_hists):
                        condition = conditions.get('background_examples_condition', 'True')
                        if not eval(condition): continue
                        if counter >= conditions['n_examples']: break
                        fig = perf_bkg.get_systematic_ratio(i, perf_nominal, show=config.get('show',False))
                        save_path = output_dir/f'systematic_ratio_background_{perf_bkg.names[i][0]}.{fmt}'
                        if fig is None:
                            continue
                        fig.savefig(str(save_path), bbox_inches='tight')
                        plt.close(fig)
                        print(f'Saved {save_path}')
                        counter+=1


        if ('bgd_shapes' in what_to_plot or 'all' in what_to_plot):
            if perf_bkg is not None:
                rng = np.random.default_rng(config["seed"])
                # Randomly sample 50 arrays
                indices = rng.choice(perf_bkg.background.shape[0], 50, replace=False)
                for arr in perf_bkg.background[indices]:
                    plt.plot(arr)
                plt.xlabel('Bin index')
                plt.ylabel('Value')
                save_path = output_dir/f'bgd_shapes.{fmt}'
                plt.savefig(str(save_path), bbox_inches='tight')
                plt.yscale('log')
                save_path = output_dir/f'bgd_shapes_log.{fmt}'
                plt.savefig(str(save_path), bbox_inches='tight')
                plt.yscale('log')
                print(f'Saved {save_path}')
                
        if ('confidence_intervals' in what_to_plot):
            if perf_sig is not None:
                fig = perf_sig.get_confidence_intervals()
                if config.get('show', False): plt.show()
                save_path = output_dir/f'confidence_intervals_signal.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')

            if perf_bkg is not None:
                fig = perf_bkg.get_confidence_intervals()
                if config.get('show', False): plt.show()
                save_path = output_dir/f'confidence_intervals_background.{fmt}'
                fig.savefig(str(save_path), bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {save_path}')
    




def main():

    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--config', help='Config file', default='configs/default.yaml')
    arg_parse.add_argument('--output_dir', help='output directory for plots')
    arg_parse.add_argument('--sig_input_dir', help='input directory with signal')
    arg_parse.add_argument('--sig_prediction_dir', help='prediction directory with signal')
    arg_parse.add_argument('--bkg_input_dir', help='input directory with bkg')
    arg_parse.add_argument('--bkg_prediction_dir', help='prediction directory with bkg')
    
    args = arg_parse.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    config = {**config["plot"], **config["raw_cuts"]}

    if args.output_dir is not None:
        config["datasets"][0]["output_dir"] = args.output_dir
    
    if args.sig_input_dir is not None:
        config["datasets"][0]["sig_input_dir"] = args.sig_input_dir
    
    if args.sig_prediction_dir is not None:
        config["datasets"][0]["sig_prediction_dir"] = args.sig_prediction_dir
    
    if args.bkg_input_dir is not None:
        config["datasets"][0]["bkg_input_dir"] = args.bkg_input_dir
    
    if args.bkg_prediction_dir is not None:
        config["datasets"][0]["bkg_prediction_dir"] = args.bkg_prediction_dir
    
    print('>>>>> Starting plotting results step...')

    plot(config)

    print('>>>>> Finished plotting results step.')

if __name__ == '__main__':
    main()
