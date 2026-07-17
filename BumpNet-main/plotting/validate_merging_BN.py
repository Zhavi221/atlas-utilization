import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import yaml
from performance_util import PerformanceUtil
from plot import get_rates, area_trapezoidal
import text_utils 
globals().update({k: v for k, v in vars(text_utils).items() if '_latex' in k})

def plot_forestValidation(config):
    # Get general configurations
    what_to_plot = config.get('what_to_plot', ['all'])
    dpi = config.get('dpi', 300)
    fmt = config.get('format', 'pdf')
    input_dir_B = Path(config['input_dir_B'])
    if not input_dir_B.exists():
        raise FileNotFoundError(f"Input directory {input_dir_B} does not exist.")
    dataset   = Path(config.get('dataset', 'func_bkg_seed_125_percent_with_signal_1.0'))
    output_dir       = Path(config.get('output_dir', './plots/forestValidation/'))
    output_dir.mkdir(parents=True, exist_ok=True)

    list_AUC = []
    list_AUC_tight = []
    list_fracCC09_SB = []
    list_deltaZmax_SB = []
    list_deltaZmax_std_SB = []
    list_deltaZmax_bin_SB = []
    list_deltaZ_std_SB = []
    list_deltaB_std_SB = []
    list_deltaZmax_std_SB_3zlr=[]
    list_deltaZmax_std_SB_5zlr=[]

    list_fracCC09_B = []
    list_deltaZmax_B = []
    list_deltaZmax_std_B = []
    list_deltaZmax_bin_B = []
    list_deltaZ_std_B = []
    list_deltaB_std_B = []
    list_deltaZmax_std_B_3zlr=[]
    list_deltaZmax_std_B_5zlr=[]

    i=0
    for (test_dir_S,test_dir_B) in zip(config['predictions_S'], config['predictions_B']):
        print(f'Processing test directory: {test_dir_S}')
        print(f'Processing test directory: {test_dir_B}')
        # Get background performance
        perf_bkg = PerformanceUtil(config,
                                    config['input_dir_B'],
                                    Path(test_dir_B),
                                    verbose=config.get('verbose', False),
                                    max_rows=config.get('max_rows', None),
                                    seed=config.get('seed', 1),
                                    shuffle=config.get('shuffle', False),
                                    edge=config.get('edge', (0.1, 1)),
                                    dpi=dpi
                                    )

        # Get S+background performance
        perf_sig = PerformanceUtil(config,
                                    config['input_dir_S'],
                                    Path(test_dir_S),
                                    verbose=config.get('verbose', False),
                                    max_rows=config.get('max_rows', None),
                                    seed=config.get('seed', 1),
                                    shuffle=config.get('shuffle', False),
                                    edge=config.get('edge', (0.1,1)),
                                    dpi=dpi
                                    )
        # store the delta Zmax information
        deltaZmax = perf_bkg.z_pred_max - perf_bkg.z_lr_max
        list_deltaZmax_B += [deltaZmax]
        list_deltaZmax_std_B += [np.std(deltaZmax)]

        deltaZmax = perf_sig.z_pred_max - perf_sig.z_lr_max
        list_deltaZmax_SB += [deltaZmax]
        list_deltaZmax_std_SB += [np.std(deltaZmax)]
        list_deltaZmax_std_SB_3zlr += [np.std(deltaZmax[perf_sig.z_lr_max>3])]
        list_deltaZmax_std_SB_5zlr += [np.std(deltaZmax[perf_sig.z_lr_max>5])]

        deltaZmax_bin_B = np.array(perf_bkg.z_pred_max_bin - perf_bkg.z_lr_max_bin)
        list_deltaZmax_bin_B += [deltaZmax_bin_B]
        deltaZmax_bin_SB = perf_sig.z_pred_max_bin - perf_sig.z_lr_max_bin
        list_deltaZmax_bin_SB += [deltaZmax_bin_SB]


        # store the delta Z information
        deltaZ = perf_bkg.z_pred - perf_bkg.z_lr
        list_deltaZ_std_B += [np.std(deltaZmax.flatten())]
        deltaZ = perf_sig.z_pred - perf_sig.z_lr
        list_deltaZ_std_SB += [np.std(deltaZmax.flatten())]


        # store the delta B information
        B = np.concatenate(perf_bkg.background)
        Bpred = np.concatenate(perf_bkg.bpred)
        SQRT = np.clip(np.sqrt(B), 1, None)
        deltaB= (B - Bpred)/SQRT
        list_deltaB_std_B += [np.std(deltaB.flatten())]

        B = np.concatenate(perf_sig.background)
        Bpred = np.concatenate(perf_sig.bpred)
        SQRT = np.clip(np.sqrt(B), 1, None)
        deltaB= (B - Bpred)/SQRT
        list_deltaB_std_SB += [np.std(deltaB.flatten())]

        # Get AUC
        zmax = np.max([np.max(x) for x in [perf_bkg.z_lr_max,
                                            perf_bkg.z_pred_max,
                                            perf_sig.z_lr_max,
                                            perf_sig.z_pred_max]])
        zmin = np.min([np.min(x) for x in [perf_bkg.z_lr_min,
                                            perf_bkg.z_pred_min,
                                            perf_sig.z_lr_min,
                                            perf_sig.z_pred_min]])

        binning = np.linspace(zmin, zmax, num=1001, endpoint=True)

        hist_pred_bkg, _ = np.histogram(perf_bkg.z_pred_max, bins=binning)
        hist_pred_sig, _ = np.histogram(perf_sig.z_pred_max, bins=binning)
        fpr_pred, tpr_pred = get_rates(hist_pred_sig, hist_pred_bkg)
        auc_pred = area_trapezoidal(fpr_pred, tpr_pred)
        list_AUC.append(auc_pred)
        print(f'AUC for {test_dir_B} : {auc_pred}')

        refined_z_pred_max_B = []
        for pred,lr in zip(perf_bkg.z_pred_max, perf_bkg.z_lr_max):
            if lr<3:refined_z_pred_max_B.append(pred)
        refined_z_pred_max_SB = []
        for pred,lr in zip(perf_sig.z_pred_max, perf_sig.z_lr_max):
            if lr>3:refined_z_pred_max_SB.append(pred)
        hist_pred_bkg, _ = np.histogram(refined_z_pred_max_B,  bins=binning)
        hist_pred_sig, _ = np.histogram(refined_z_pred_max_SB, bins=binning)
        fpr_pred, tpr_pred = get_rates(hist_pred_sig, hist_pred_bkg)
        auc_pred_tight = area_trapezoidal(fpr_pred, tpr_pred)
        list_AUC_tight.append(auc_pred_tight)
        print(f'Tight AUC for {test_dir_B} : {auc_pred_tight}')
    
        # Get fracCC09
        def get_fracCC09(z_pred_corrCoeff):
            return (z_pred_corrCoeff > 0.9).sum() / z_pred_corrCoeff.shape[0]

        list_fracCC09_B.append(get_fracCC09(perf_bkg.z_pred_corrCoeff))
        list_fracCC09_SB.append(get_fracCC09(perf_sig.z_pred_corrCoeff))

        i+=1
        #if i>1: break

    perf_bkg = PerformanceUtil(config,
                                config['input_dir_B'],
                                Path(config['predictions_merge_B']),
                                verbose=config.get('verbose', False),
                                max_rows=config.get('max_rows', None),
                                seed=config.get('seed', 1),
                                shuffle=config.get('shuffle', False),
                                edge=config.get('edge', (0.1, 1)),
                                dpi=dpi
                                )

    # Get S+background performance
    perf_sig = PerformanceUtil(config,
                                config['input_dir_S'],
                                Path(config['predictions_merge_S']),
                                verbose=config.get('verbose', False),
                                max_rows=config.get('max_rows', None),
                                seed=config.get('seed', 1),
                                shuffle=config.get('shuffle', False),
                                edge=config.get('edge', (0.1,1)),
                                dpi=dpi
                                )
    # store the delta Zmax information
    deltaZmax = perf_bkg.z_pred_max - perf_bkg.z_lr_max
    deltaZmax_std_B = np.std(deltaZmax)

    deltaZmax = perf_sig.z_pred_max - perf_sig.z_lr_max
    deltaZmax_std_SB = np.std(deltaZmax)
    deltaZmax_std_SB_3zlr = np.std(deltaZmax[perf_sig.z_lr_max>3])
    deltaZmax_std_SB_5zlr = np.std(deltaZmax[perf_sig.z_lr_max>5])

    # store the delta Z information
    deltaZ = perf_bkg.z_pred - perf_bkg.z_lr
    deltaZ_std_B = np.std(deltaZmax.flatten())
    deltaZ = perf_sig.z_pred - perf_sig.z_lr
    deltaZ_std_SB = np.std(deltaZmax.flatten())


    # store the delta B information
    B = np.concatenate(perf_bkg.background)
    Bpred = np.concatenate(perf_bkg.bpred)
    SQRT = np.clip(np.sqrt(B), 1, None)
    deltaB= (B - Bpred)/SQRT
    deltaB_std_B = np.std(deltaB.flatten())

    B = np.concatenate(perf_sig.background)
    Bpred = np.concatenate(perf_sig.bpred)
    SQRT = np.clip(np.sqrt(B), 1, None)
    deltaB= (B - Bpred)/SQRT
    deltaB_std_SB = np.std(deltaB.flatten())

    # Get AUC
    zmax = np.max([np.max(x) for x in [perf_bkg.z_lr_max,
                                        perf_bkg.z_pred_max,
                                        perf_sig.z_lr_max,
                                        perf_sig.z_pred_max]])
    zmin = np.min([np.min(x) for x in [perf_bkg.z_lr_min,
                                        perf_bkg.z_pred_min,
                                        perf_sig.z_lr_min,
                                        perf_sig.z_pred_min]])

    binning = np.linspace(zmin, zmax, num=1001, endpoint=True)

    hist_pred_bkg, _ = np.histogram(perf_bkg.z_pred_max, bins=binning)
    hist_pred_sig, _ = np.histogram(perf_sig.z_pred_max, bins=binning)
    fpr_pred, tpr_pred = get_rates(hist_pred_sig, hist_pred_bkg)
    auc_pred = area_trapezoidal(fpr_pred, tpr_pred)

    refined_z_pred_max_B = []
    for pred,lr in zip(perf_bkg.z_pred_max, perf_bkg.z_lr_max):
        if lr<3:refined_z_pred_max_B.append(pred)
    refined_z_pred_max_SB = []
    for pred,lr in zip(perf_sig.z_pred_max, perf_sig.z_lr_max):
        if lr>3:refined_z_pred_max_SB.append(pred)
    hist_pred_bkg, _ = np.histogram(refined_z_pred_max_B,  bins=binning)
    hist_pred_sig, _ = np.histogram(refined_z_pred_max_SB, bins=binning)
    fpr_pred, tpr_pred = get_rates(hist_pred_sig, hist_pred_bkg)
    auc_pred_tight = area_trapezoidal(fpr_pred, tpr_pred)
    print(f'Tight AUC for {test_dir_B} : {auc_pred_tight}')

    # Get fracCC09
    fracCC09_B = (perf_bkg.z_pred_corrCoeff > 0.9).sum() / perf_bkg.z_pred_corrCoeff.shape[0]
    fracCC09_SB = (perf_sig.z_pred_corrCoeff > 0.9).sum() / perf_sig.z_pred_corrCoeff.shape[0]


    #get std and mean
    list_deltaZmax_bin_B  =  np.vstack([np.array(x, dtype=float) for x in list_deltaZmax_bin_B])
    list_deltaZmax_bin_SB =  np.vstack([np.array(x, dtype=float) for x in list_deltaZmax_bin_SB])
    list_deltaZmax_B  =  np.vstack([np.array(x, dtype=float) for x in list_deltaZmax_B])
    list_deltaZmax_SB =  np.vstack([np.array(x, dtype=float) for x in list_deltaZmax_SB])

    mean_deltaZmax_bin_B  = np.mean(list_deltaZmax_bin_B, axis=0)
    std_deltaZmax_bin_B   = np.std(np.array(list_deltaZmax_bin_B), axis=0)
    mean_deltaZmax_bin_SB = np.mean(list_deltaZmax_bin_SB, axis=0)
    std_deltaZmax_bin_SB  = np.std(list_deltaZmax_bin_SB, axis=0)

    mean_deltaZmax_B  = np.mean(list_deltaZmax_B, axis=0)
    std_deltaZmax_B   = np.std(list_deltaZmax_B, axis=0)
    mean_deltaZmax_SB = np.mean(list_deltaZmax_SB, axis=0)
    std_deltaZmax_SB  = np.std(list_deltaZmax_SB, axis=0)


    if 'deltaZmax' in what_to_plot or 'all' in what_to_plot:
        plt.figure(figsize=(6,5), dpi=dpi)
        plt.hist(std_deltaZmax_B, bins=50, alpha=0.5, label=f'B-only (<sigma>={np.mean(std_deltaZmax_B):.2g}+/-{np.std(std_deltaZmax_B):.2g})', color='blue', density=True)
        plt.hist(std_deltaZmax_SB, bins=50, alpha=0.5, label=f'S+B (<sigma>={np.mean(std_deltaZmax_SB):.2g}+/-{np.std(std_deltaZmax_SB):.2g})', color='red', density=True)
        plt.xlabel(sigma_delta_zmax_latex, fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend(fontsize=12)
        plt.savefig(output_dir / f'deltaZmax_sigma.{fmt}', format=fmt, dpi=dpi)
        print(f'Saved {output_dir / f"deltaZmax_sigma.{fmt}"}')
        plt.close()

        plt.figure(figsize=(6,5), dpi=dpi)
        plt.hist(mean_deltaZmax_B, bins=50, alpha=0.5, label=f'B-only (<mu>={np.mean(mean_deltaZmax_B):.2g}+/-{np.std(mean_deltaZmax_B):.2g})', color='blue', density=True)
        plt.hist(mean_deltaZmax_SB, bins=50, alpha=0.5, label=f'S+B (<mu>={np.mean(mean_deltaZmax_SB):.2g}+/-{np.std(mean_deltaZmax_SB):.2g})', color='red', density=True)
        plt.xlabel(mean_zmax_latex, fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend(fontsize=12)
        plt.savefig(output_dir / f'deltaZmax_mean.{fmt}', format=fmt, dpi=dpi)
        print(f'Saved {output_dir / f"deltaZmax_mean.{fmt}"}')
        plt.close()

        plt.figure(figsize=(6,5), dpi=dpi)
        plt.hist(std_deltaZmax_bin_B, bins=50, alpha=0.5, label=f'B-only (<sigma>={np.mean(std_deltaZmax_bin_B):.3g}+/-{np.std(std_deltaZmax_bin_B):.2g})', color='blue', density=True)
        plt.hist(std_deltaZmax_bin_SB, bins=50, alpha=0.5, label=f'S+B (<sigma>={np.mean(std_deltaZmax_bin_SB):.3g}+/-{np.std(std_deltaZmax_bin_SB):.2g})', color='red', density=True)
        plt.xlabel(r"$\sigma(\Delta Z_{\mathrm{max}}$ bin)", fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend(fontsize=12)
        plt.savefig(output_dir / f'deltaZmax_bin_sigma.{fmt}', format=fmt, dpi=dpi)
        print(f'Saved {output_dir / f"deltaZmax_bin_sigma.{fmt}"}')
        plt.close()
        plt.figure(figsize=(6,5), dpi=dpi)
        plt.hist(std_deltaZmax_bin_B, bins=50, alpha=0.5, label=f'B-only (<sigma>={np.mean(std_deltaZmax_bin_B):.3g}+/-{np.std(std_deltaZmax_bin_B):.2g})', range=[0, 1],color='blue', density=True)
        plt.hist(std_deltaZmax_bin_SB, bins=50, alpha=0.5, label=f'S+B (<sigma>={np.mean(std_deltaZmax_bin_SB):.3g}+/-{np.std(std_deltaZmax_bin_SB):.2g})',  range=[0, 1], color='red', density=True)
        plt.xlabel(r"$\sigma(\Delta Z_{\mathrm{max}}$ bin)", fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend(fontsize=12)
        plt.savefig(output_dir / f'deltaZmax_bin_sigma_zoom.{fmt}', format=fmt, dpi=dpi)
        print(f'Saved {output_dir / f"deltaZmax_bin_sigma_zoom.{fmt}"}')
        plt.close()

        plt.figure(figsize=(6,5), dpi=dpi)
        plt.hist(mean_deltaZmax_bin_B, bins=50, alpha=0.5, label=f'B-only (<mu>={np.mean(mean_deltaZmax_bin_B):.2g}+/-{np.std(mean_deltaZmax_bin_B):.2g})', color='blue', density=True)
        plt.hist(mean_deltaZmax_bin_SB, bins=50, alpha=0.5, label=f'S+B (<mu>={np.mean(mean_deltaZmax_bin_SB):.2g}+/-{np.std(mean_deltaZmax_bin_SB):.2g})', color='red', density=True)
        plt.xlabel(r"$\mu(\Delta Z_{\mathrm{max}}$ bin)", fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend(fontsize=12)
        plt.savefig(output_dir / f'deltaZmax_bin_mean.{fmt}', format=fmt, dpi=dpi)
        print(f'Saved {output_dir / f"deltaZmax_bin_mean.{fmt}"}')
        plt.close()

        # deltazmax parameters vs perf_sig.z_lr_max
        # mean  
        plt.figure(figsize=(6,5), dpi=dpi)
        plt.hist2d(perf_sig.z_lr_max, mean_deltaZmax_SB, bins=50, cmap='Reds')
        plt.xlabel(z_true_max_latex, fontsize=14)
        plt.ylabel(mean_zmax_latex, fontsize=14)
        plt.savefig(output_dir / f'deltaZmax_mean_vs_ztrue_max_SBonly.{fmt}', format=fmt, dpi=dpi)
        print(f'Saved {output_dir / f"deltaZmax_mean_vs_ztrue_max_SBonly.{fmt}"}')
        plt.close()

        plt.figure(figsize=(6,5), dpi=dpi)
        plt.hist2d(perf_bkg.z_lr_max, mean_deltaZmax_B, bins=50, cmap='Reds')
        plt.xlabel(z_true_max_latex, fontsize=14)
        plt.ylabel(mean_zmax_latex, fontsize=14)
        plt.savefig(output_dir / f'deltaZmax_mean_vs_ztrue_max_Bonly.{fmt}', format=fmt, dpi=dpi)
        print(f'Saved {output_dir / f"deltaZmax_mean_vs_ztrue_max_Bonly.{fmt}"}')
        plt.close()

        # std
        plt.figure(figsize=(6,5), dpi=dpi)
        plt.hist2d(perf_sig.z_lr_max, std_deltaZmax_SB, bins=50, cmap='Reds')
        plt.xlabel(z_true_max_latex, fontsize=14)
        plt.ylabel(sigma_delta_zmax_latex, fontsize=14)
        plt.savefig(output_dir / f'deltaZmax_sigma_vs_ztrue_max_SBonly.{fmt}', format=fmt, dpi=dpi)
        print(f'Saved {output_dir / f"deltaZmax_sigma_vs_ztrue_max_SBonly.{fmt}"}')
        plt.close()

        plt.figure(figsize=(6,5), dpi=dpi)
        plt.hist2d(perf_bkg.z_lr_max, std_deltaZmax_B, bins=50, cmap='Reds')
        plt.xlabel(z_true_max_latex, fontsize=14)
        plt.ylabel(sigma_delta_zmax_latex, fontsize=14)
        plt.savefig(output_dir / f'deltaZmax_sigma_vs_ztrue_max_Bonly.{fmt}', format=fmt, dpi=dpi)
        print(f'Saved {output_dir / f"deltaZmax_sigma_vs_ztrue_max_Bonly.{fmt}"}')
        plt.close()

    if 'AUC' in what_to_plot or 'all' in what_to_plot:
        plt.figure(figsize=(6,5), dpi=dpi)
        plt.hist([list_AUC, [auc_pred]], bins=50, alpha=0.7, color=['green', 'red'], density=False, label=[f'Single training (<AUC>={np.mean(list_AUC):.4g}+/-{np.std(list_AUC):.2g})', 'Averaged'])
        plt.xlabel('AUC', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend()
        plt.savefig(output_dir / f'AUC_distribution.{fmt}', format=fmt, dpi=dpi)
        print(f'Saved {output_dir / f"AUC_distribution.{fmt}"}')
        plt.close()

        plt.figure(figsize=(6,5), dpi=dpi)
        plt.hist([list_AUC_tight, [auc_pred_tight]], bins=50, alpha=0.7, color=['green', 'red'], density=False, label=[f'Single training (<AUC>={np.mean(list_AUC_tight):.4g}+/-{np.std(list_AUC_tight):.2g})', 'Averaged'])
        plt.xlabel('AUC (threshold at 3)', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend()
        plt.savefig(output_dir / f'AUCsam_distribution.{fmt}', format=fmt, dpi=dpi)
        print(f'Saved {output_dir / f"AUCsan_distribution.{fmt}"}')
        plt.close()

    if 'fracCC09' in what_to_plot or 'all' in what_to_plot:
        ##scatter plot
        plt.figure(figsize=(6,5), dpi=dpi)
        plt.scatter(list_fracCC09_B, list_fracCC09_SB, color='green', label='Single training')
        plt.scatter(    [fracCC09_B],    [fracCC09_SB],color='red', label='Averaged')
        plt.xlabel('Fraction of events with CorrCoeff > 0.9 (B only)', fontsize=14)
        plt.ylabel('Fraction of events with CorrCoeff > 0.9 (S+B)', fontsize=14)
        plt.legend()
        #plt.xlim(0,1)
        #plt.ylim(0,1)
        #plt.plot([0,1],[0,1], color='black', linestyle='----', linewidth=0.5)
        plt.savefig(output_dir / f'fracCC09_scatter.{fmt}', format=fmt, dpi=dpi)
        print(f'Saved {output_dir / f"fracCC09_scatter.{fmt}"}')
        plt.close()

    if 'deltaZ_vs_deltaB' in  what_to_plot or 'all' in what_to_plot:

        def produce_scatter_plot(list_x, list_y, merged_x, merged_y, BorSB, label_x, label_y, outName, fmt, dpi):
            plt.figure(figsize=(6,5), dpi=dpi)
            plt.scatter(list_x,   list_y, color='green', label=BorSB, marker=',')
            plt.scatter([merged_x],   [merged_y], color='red', label=BorSB+', Averaged', marker=',')
            plt.xlabel(label_x, fontsize=14)
            plt.ylabel(label_y, fontsize=14)
            plt.legend()
            plt.savefig(outName, format=fmt, dpi=dpi)
            print(f'Saved {outName}')
            plt.close()

        produce_scatter_plot(list_deltaB_std_B,  list_deltaZ_std_B,  deltaB_std_B,  deltaZ_std_B,    'B', sigma_delta_b_latex, sigma_delta_z_latex, output_dir / f'deltaZ_vs_deltaB_B.{fmt}',  fmt, dpi)
        produce_scatter_plot(list_deltaB_std_SB, list_deltaZ_std_SB, deltaB_std_SB, deltaZ_std_SB, 'S+B', sigma_delta_b_latex, sigma_delta_z_latex,output_dir / f'deltaZ_vs_deltaB_SB.{fmt}', fmt, dpi)
        produce_scatter_plot(list_deltaB_std_B,  list_deltaZmax_std_B,  deltaB_std_B,  deltaZmax_std_B,    'B', sigma_delta_b_latex, sigma_delta_zmax_latex, output_dir / f'deltaZmax_vs_deltaB_B.{fmt}',  fmt, dpi)
        produce_scatter_plot(list_deltaB_std_SB, list_deltaZmax_std_SB, deltaB_std_SB, deltaZmax_std_SB, 'S+B', sigma_delta_b_latex, sigma_delta_zmax_latex, output_dir / f'deltaZmax_vs_deltaB_SB.{fmt}', fmt, dpi)
        produce_scatter_plot(list_AUC, list_deltaZmax_std_B,  auc_pred, deltaZmax_std_B,    'B', 'AUC', sigma_delta_zmax_latex, output_dir / f'deltaZmax_vs_AUC_B.{fmt}',  fmt, dpi)
        produce_scatter_plot(list_AUC, list_deltaZmax_std_SB, auc_pred, deltaZmax_std_SB, 'S+B', 'AUC', sigma_delta_zmax_latex, output_dir / f'deltaZmax_vs_AUC_SB.{fmt}', fmt, dpi)
        produce_scatter_plot(list_AUC, list_deltaZmax_std_SB_5zlr, auc_pred, deltaZmax_std_SB_5zlr, 'S+B', 'AUC', sigma_delta_zmax_latex+' (zlr>5)', output_dir / f'deltaZmax_ZLR5_vs_AUC_SB.{fmt}', fmt, dpi)

    
        plt.figure(figsize=(6,5), dpi=dpi)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)
        plt.scatter(list_deltaZmax_std_SB_3zlr, list_deltaZmax_std_SB_5zlr, color='green', label='S+B')
        plt.scatter([deltaZmax_std_SB_3zlr], [deltaZmax_std_SB_5zlr], color='red', label='S+B, Averaged')
        plt.xlabel(sigma_delta_zmax_latex+' [zlr>3]', fontsize=14)
        plt.ylabel(sigma_delta_zmax_latex+' [zlr>5]', fontsize=14)
        plt.legend()
        plt.text(0.15, 0.15, sigma_delta_zmax_latex+f' [zlr>3] : Averaged / mean={deltaZmax_std_SB_3zlr/np.mean(list_deltaZmax_std_SB_3zlr):.1%}',transform=ax.transAxes)
        plt.text(0.15, 0.1 , sigma_delta_zmax_latex+f' [zlr>3] : Averaged / best={deltaZmax_std_SB_3zlr/np.min(list_deltaZmax_std_SB_3zlr):.1%}',transform=ax.transAxes)
        plt.text(0.1 , 0.8,  sigma_delta_zmax_latex+f' [zlr>5] : Averaged / mean={deltaZmax_std_SB_5zlr/np.mean(list_deltaZmax_std_SB_5zlr):.1%}',transform=ax.transAxes)
        plt.text(0.1 , 0.75,  sigma_delta_zmax_latex+f' [zlr>5] : Averaged / best={deltaZmax_std_SB_5zlr/np.min(list_deltaZmax_std_SB_5zlr):.1%}',transform=ax.transAxes)
        plt.savefig(output_dir / f'deltaZmax_ZLR5_vs_deltaZmax_ZLR3_SB.{fmt}', format=fmt, dpi=dpi)
        print(f'Saved {output_dir / f"deltaZmax_ZLR5_vs_deltaZmax_ZLR3_SB.{fmt}"}')
        plt.close()

def main():

    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--config', help='Config file', default='configs/default.yaml')
    arg_parse.add_argument('--output_dir', help='output directory')
    arg_parse.add_argument('--input_dir_S', help='input directory')
    arg_parse.add_argument('--input_dir_B', help='input directory')
    arg_parse.add_argument('--predictions_S', help='list of training to be merged', nargs='*')
    arg_parse.add_argument('--predictions_B', help='list of training to be merged', nargs='*')
    arg_parse.add_argument('--predictions_merge_S', help='training after merging')
    arg_parse.add_argument('--predictions_merge_B', help='training after merging')
     
    args = arg_parse.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    config = {**config["validateMergingBN"], **config["raw_cuts"]}

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    
    if args.input_dir_S is not None:
        config["input_dir_S"] = args.input_dir_S

    if args.input_dir_B is not None:
        config["input_dir_B"] = args.input_dir_B
    
    if args.predictions_S is not None:
        config["predictions_S"] = args.predictions_S
    
    if args.predictions_B is not None:
        config["predictions_B"] = args.predictions_B
    
    if args.predictions_merge_S is not None:
        config["predictions_merge_S"] = args.predictions_merge_S
    
    if args.predictions_merge_B is not None:
        config["predictions_merge_B"] = args.predictions_merge_B
    
    
    
    print(f'Producing validation plots')
    plot_forestValidation(config)
    print(f'Producing validation plots............ Done !')


if __name__ == '__main__':
    main()    
