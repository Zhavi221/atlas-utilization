import numpy as np
import re
from collections import Counter

def replace_signature_with_latex(signature, dataset):
    '''Replace signature in the histogram name with corresponding LaTeX'''

    if dataset == 'DM':
        object_patterns = {
            r'$\gamma$':r'_\d+g',
            r'$e$':r'_\d+e',
            r'$\mu$':r'_\d+m',
            r'Wh':r'_\d+Wh',
            r'T':r'_\d+T',
            r'HM':r'_\d+HM',
            r'Z':r'_\d+Z',
            r'b':r'_\d+bExc',
            r'j':r'_\d+ex',
            r'+j':r'_\d+in',
        }
    if dataset == 'ATLAS':
        object_patterns = {
            r'$\gamma$':r'_\d+gx',
            r'$e$':r'\d+ex',
            r'$\mu$':r'_\d+mx',
            r'j':r'_\d+jx',
            r'b':r'_\d+bx',
            r'Z':r'_\d+Zx',
        }
    objects = {}
    for obj, pattern in object_patterns.items():
        element = re.findall(pattern, signature)
        if len(element) == 0:
            continue

        if obj == 'Wh': obj = '$V_{h}$'
        objects[obj] = re.findall(r'\d+', element[0])[0]

    output = ' + '.join([f'{n}{obj}'  for obj, n in objects.items() if n != '0'])

    cuts = {'OS' : 'OS',
            'SS' : 'SS',
            r"(\d+)met": "$E_\\mathrm{T}^{\\mathrm{miss}} > VALUE \\mathrm{GeV}$",
            r"(\d+)E0pt": "$p_\\mathrm{T}(e_{0}) > VALUE \\mathrm{GeV}$",
            r"(\d+)M0pt": "$p_\\mathrm{T}(\\mu_{0}) > VALUE \\mathrm{GeV}$",
            r"(\d+)G0pt": "$p_\\mathrm{T}(\\gamma_{0}) > VALUE \\mathrm{GeV}$",
            r"(\d+)Wh0pt": "$p_\\mathrm{T}(V_{h0}) > VALUE \\mathrm{GeV}$",
            r"(\d+)Z0pt": "$p_\\mathrm{T}(Z_{0}) > VALUE \\mathrm{GeV}$",
            r"(\d+)T0pt": "$p_\\mathrm{T}(T_{0}) > VALUE \\mathrm{GeV}$",
            r"(\d+)HM0pt": "$p_\\mathrm{T}(HM_{0}) > VALUE \\mathrm{GeV}$"
            }

    for pattern, latex in cuts.items():
        match = re.findall(pattern, signature)
        if len(match) != 0:
            output += ', '
            output += latex.replace('VALUE', match[0])

    return output

def get_signature_and_observable(histogram_name):
    '''Gets the histogram final state and observable from its name.
    Returns strings in LaTeX for the signature and observable, respectively.'''

    if isinstance(histogram_name, np.ndarray):
        histogram_name = histogram_name[0]

    # Histograms from analytical functions
    if 'hCat' not in histogram_name:
        functions = {
            'linear' : '$p_0 x + p_1$',
            'exponential' : '$p_0 e^{-p_1} x$',
            'one_over_x' : r'$\frac{1}{p_0 x} + p_1$',
            'one_over_x_squared' : r'$\frac{1}{p_0 x^{2}} + p_1$',
            'one_over_x_cubed' : r'$\frac{1}{p_0 x^{3}} + p_1$',
            'one_over_x_to_4th' : r'$\frac{1}{p_0 x^{4}} + p_1$',
            'one_over_x_to_nth' : r'$\frac{1}{p_0 x^{n}} + p_1, n \in [0.01, 10]$',
            'parabola_half' : r'$-p_0 (x - x_{max})^{2} + y_{min}$',
            'ln_negative' : r'$-p_0 \ln(x) + p_1$',
            'cos_quarter' : r'$\Delta y \cos[p_0 (x - p_1)] + y_{max}$',
            'cosh_half' : r'$\cosh[p_0(x - x_{max})] + p_1$'
        }
        for pattern, function in functions.items():
            element = re.findall(pattern, histogram_name)
            if len(element) == 0: continue
            signature = function + ',\n'
            parameters = histogram_name.replace(pattern+'_','').split('_')
            signature += f'$p_0$={parameters[0]}, $p_1$={parameters[1]}'
        observable = '$m$'

    # Histograms from DarkMachines
    if 'hCat' in histogram_name:
        # Signature
        signature = histogram_name.split('__')[0].replace('hCat','')
        signature = replace_signature_with_latex(signature, dataset='DM')

        # Observable
        observable = ''
        if len(re.findall('massMET', histogram_name)) != 0:
            observable += r'$m(E_{\mathrm{T}}^{\mathrm{miss}},'
        elif len(re.findall('mass', histogram_name)) != 0:
            observable += r'$m('
        else:
            raise Exception('Could not identify observable in histogram name.')
        combination = histogram_name.split('__')[1].split('_')[0]
        replacements = {'.':',', 'el':'e', 'mu':'\\mu', 'top':'T', 'Wh':'V_{h}'}
        for str_in, str_out in replacements.items():
            combination = combination.replace(str_in, str_out)
        observable += combination
        observable += ')$'

    # Histograms from ATLAS
    if 'cat' in histogram_name:
        # Signature
        signature = histogram_name.split('cat', 1)[1].lstrip('_')
        signature = replace_signature_with_latex(signature, dataset='ATLAS')

        # Observable
        observable = ''
        if len(re.findall('massMET', histogram_name)) != 0:
            observable += r'$m(E_{\mathrm{T}}^{\mathrm{miss}},'
        elif len(re.findall('mass', histogram_name)) != 0:
            observable += r'$m('
        else:
            raise Exception('Could not identify observable in histogram name.')
        combination = histogram_name.split('_')[1]
        replacements = {'.':',', 'm':'\\mu', 'g':'\\gamma'}
        for str_in, str_out in replacements.items():
            combination = combination.replace(str_in, str_out)
        observable += combination
        observable += ')$'

    return signature, observable

def parse_observable(hist_name: str):
    """Parse observable part (between 'mass_' and '_cat_')."""
    observable = hist_name.split("mass_")[1].split("_cat_")[0]
    number_of_objects = [o for o in observable if o.isalpha()]
    return dict(Counter(number_of_objects))

def parse_final_state(hist_name: str):
    """Parse final state part (after 'cat_')."""
    final_state = hist_name.split("cat_")[1]
    # strip SS/OS tag if present
    final_state = re.sub(r"_(SS|OS)$", "", final_state)
    matches = re.findall(r"(\d+)(ex|mx|bx|jx|Zx)", final_state)
    return {k: int(v) for v, k in matches}

def parse_charge_tag(hist_name: str):
    """Return 'SS', 'OS', or None depending on the histogram name."""
    m = re.search(r"(SS|OS)$", hist_name)
    return m.group(1) if m else None

def parse_condition(cond: str):
    """Convert string like '>=2' → (op, value)."""
    match = re.match(r"(>=|<=|==|>|<)?(\d+)", cond.strip())
    if not match:
        raise ValueError(f"Invalid condition format: {cond}")
    op, val = match.groups()
    return (op or "=="), int(val)

def eval_condition(n: int, op: str, val: int) -> bool:
    """Evaluate condition."""
    if op == "==": return n == val
    if op == ">=": return n >= val
    if op == ">":  return n > val
    if op == "<=": return n <= val
    if op == "<":  return n < val
    raise ValueError(f"Unsupported operator {op}")

def passes_selection(hist_name: str, selection: dict):
    """
    Check observable, final state, and charge tag.

    Parameters:
    -----------
    hist_name : str
        Histogram name to evaluate. Example:
        'mass_e0j0_cat_2ex_0mx_0bx_2jx_0Zx'
    selection : dict
        Selection criteria dictionary. Numbers represent the amount of desired objects. 
        Example: to select observables with 2 electrons and 1 jet from the 2e + >=1j 
        final states, the condition would be
        selection:
            observable:
                e: "==2"
                j: "==1"
            final_state:
                ex: "==2"
                jx: ">=1"
            charge: "OS"   # can be "SS", "OS", or omitted

    Returns:
    --------
    bool
        True if histogram passes selection, False otherwise.
    """
    
    # Selection criteria does not apply to functions
    if 'cat' not in hist_name:
        return True

    obs_counts = parse_observable(hist_name)
    fs_counts  = parse_final_state(hist_name)
    charge_tag = parse_charge_tag(hist_name)

    # Check observable
    for obj, cond in selection.get("observable", {}).items():
        op, val = parse_condition(cond)
        n = obs_counts.get(obj, 0)
        if not eval_condition(n, op, val):
            return False

    # Check final state
    for obj, cond in selection.get("final_state", {}).items():
        op, val = parse_condition(cond)
        n = fs_counts.get(obj, 0)
        if not eval_condition(n, op, val):
            return False

    # Check charge tag
    if "charge" in selection:
        required = selection["charge"]
        # if hist has SS/OS, must match
        # if hist has no tag, let it pass
        if charge_tag is not None and charge_tag != required:
            return False

    return True
