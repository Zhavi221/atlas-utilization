import numpy as np
import os
from tqdm import tqdm

def smooth_MC_bumps_neighbor(hist):
    
    nbins = hist.GetNbinsX()
        
    for ibin in range(24, nbins + 1):
        content = hist.GetBinContent(ibin)
        error = hist.GetBinError(ibin)

        # get neighbors (avoid under/overflow bins)
        left = hist.GetBinContent(ibin - 1) if ibin > 1 else content
        left_error = hist.GetBinError(ibin - 1) if ibin > 1 else error
        
        # check if bin content is suspiciously large
        if ibin != 0 and (content > (left + 3*left_error)) and content > 1:

            right_bin = ibin+1 if ibin+1 < nbins else ibin
            right = hist.GetBinContent(right_bin)

            while right > (left + 3*left_error) and right_bin < nbins:
                right = hist.GetBinContent(right_bin)
                right_error = hist.GetBinError(right_bin)
                right_bin += 1

            print(f"Anomal bump detected at {ibin} for {hist.GetName()}")
            # replace bin content with neighbor average
            new_content = 0.5 * (left + right)
            hist.SetBinContent(ibin, new_content)

            # reset error to the average
            new_error = 0.5 * (left - new_content)
            hist.SetBinError(ibin, new_error)

    return hist

def load_ATLAS(dir_path, cuts=None, MC=False):
    """
    Loads ATLAS ROOT file and extracts histogram names, entries (all, SM, and BSM), bin edges, and errors.

    Parameters:
    -----------
    dir_path : str
    cuts : dict, optional. YAML file should include something like the following:
        raw_cuts:
            min_num_bins: 25
            min_total_events: 100
            min_num_events: 0.35
            skipped_bins: 0
    MC : bool, optional
        If True, applies smooth_MC_bumps_neighbor to MC histograms to remove bumps.

    Returns:
    --------
    data : dict
        Dictionary containing histogram data with keys 'bin_content', 'bin_content_bsm', 'bin_edges', 'bin_errors', and 'names'.
    """
    import ROOT

    # Define SM processes
    sm_processes = ['VV_roi', 'Zee_roi', 'ttbar_roi', 'ttX_roi', 'singletop_roi', 'Wlnu_roi', 'Zmumu_roi', 'Wtaunu_roi', 'ttXX_roi', 'Ztautau_roi']

    if not os.path.exists(dir_path): raise FileNotFoundError(f"File not found: {dir_path}")
    tfile = ROOT.TFile.Open(f'{dir_path}' ,"READ")
    hist_names = [key.GetName() for key in tfile.GetListOfKeys()]

    bin_content, bin_content_bsm, bin_content_sm, names, bin_edges, bin_errors = [], [], [], [], [], []

    for hist_name in tqdm(hist_names, desc='Loading histograms', total=len(hist_names)):
        if not hist_name.startswith('ROI'): continue

        hist = tfile.Get(f'{hist_name}')
        if not hist: print(f"Failed to get histogram {hist_name}")

        if isinstance(hist, ROOT.THStack):
            
            # Convert THStack to TH1 by summing its histograms
            unstacked_hists = hist.GetHists()
            if not unstacked_hists or unstacked_hists.GetSize() == 0: print(f"THStack {hist_name} is empty")

            stacked_hist_all = unstacked_hists.At(0).Clone("stacked_hist_all")
            stacked_hist_bsm = unstacked_hists.At(0).Clone("stacked_hist_bsm")
            stacked_hist_all.Reset()
            stacked_hist_bsm.Reset()

            for i in range(unstacked_hists.GetSize()):
                hist_i = unstacked_hists.At(i)
                if not hist_i: continue
                if hist_i.GetName() not in sm_processes: stacked_hist_bsm.Add(hist_i)
                stacked_hist_all.Add(hist_i)

        elif isinstance(hist, ROOT.TH1):

            stacked_hist_all = hist

            # Create empty clones for SM and BSM so nothing breaks later
            stacked_hist_bsm = hist.Clone("stacked_hist_bsm")
            stacked_hist_bsm.Reset()

        hist_name = hist_name.replace('rebinned_', '') 
        hist_name = hist_name.replace('ROI_', '') 

        # Create SM histogram by subtracting BSM from total
        stacked_hist_sm = stacked_hist_all.Clone("stacked_hist_sm")
        stacked_hist_sm.Add(stacked_hist_bsm, -1)

        if MC and stacked_hist_sm.Integral() != 0:
            stacked_hist_sm = smooth_MC_bumps_neighbor(stacked_hist_sm)
            
            # After smoothing SM, rebuild total histogram from SM + BSM so the smoothed SM is reflected
            stacked_hist_all.Reset()
            stacked_hist_all.Add(stacked_hist_sm)
            stacked_hist_all.Add(stacked_hist_bsm)

        # Obtain entries, edges, and errors
        nbins = stacked_hist_all.GetNbinsX()
        bin_content_i = np.array([stacked_hist_all.GetBinContent(b+1) for b in range(nbins)])
        bin_content_bsm_i = np.array([stacked_hist_bsm.GetBinContent(b+1) for b in range(nbins)])
        bin_edges_i = np.array([stacked_hist_all.GetBinLowEdge(b+1) for b in range(nbins)] + [stacked_hist_all.GetBinLowEdge(nbins+1)])
        bin_errors_i = np.array([stacked_hist_all.GetBinError(b) for b in range(1, nbins+1)])

        # Apply the cuts
        if cuts is not None:
            try:
                # Remove data after last bin with 'min_num_events' events
                low_stats_cut = np.where(bin_content_i > cuts["min_num_events"])[0]

                if "skipped_bins" in cuts: 
                    start_bin = int(cuts["skipped_bins"]*len(bin_content_i))
                else:
                    start_bin = 0

                bin_content_i = bin_content_i[start_bin:low_stats_cut[-1]+1]
                bin_content_bsm_i = bin_content_bsm_i[start_bin:low_stats_cut[-1]+1]
                bin_edges_i = bin_edges_i[start_bin:low_stats_cut[-1]+2]
                bin_errors_i = bin_errors_i[start_bin:low_stats_cut[-1]+1]
            except:
                continue

            # Enforce bin cut
            if len(bin_content_i) < cuts["min_num_bins"]: 
                continue
            if sum(bin_content_i) < cuts["min_total_events"]: 
                continue
            
        if  bin_content_i[0] > 0 and len(bin_content_i) > 0 :
            bin_content.append(bin_content_i)
            bin_content_bsm.append(bin_content_bsm_i)
            names.append(hist_name)
            bin_edges.append(bin_edges_i)
            bin_errors.append(bin_errors_i)

    data = {
        'bin_content': np.array(bin_content, dtype=object),
        'bin_content_bsm': np.array(bin_content_bsm, dtype=object),
        'names': np.array(names, dtype=object),
        'bin_edges': np.array(bin_edges, dtype=object),
        'bin_errors': np.array(bin_errors, dtype=object),
    }

    tfile.Close()
    print(f'> Successfully loaded {len(data["bin_content"])} histograms from {dir_path}')
    return data