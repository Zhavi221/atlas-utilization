from scipy.interpolate import CubicSpline

def calc_n_derivs_avg(x, y, n_derivs):
    if len(x) < 2 or len(y) < 2:
        return ["Can't plot, not enough data points."] * n_derivs
    
    cs = CubicSpline(x, y)
    derivs_averages = []
    for n in range(n_derivs):
        n_deriv = cs(x, n)
        deriv_average = sum(n_deriv)//len(n_deriv)
        
        derivs_averages.append(deriv_average)

    return derivs_averages