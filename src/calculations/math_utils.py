from scipy.interpolate import CubicSpline

def calc_n_derivs_avg(x, y, n_derivs):
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return ["Can't plot, not enough data points."] * n_derivs

    # ensure x is strictly increasing because CubicSpline requires it
    try:
        increasing = all(x[i] < x[i + 1] for i in range(len(x) - 1))
    except Exception:
        xl = list(x)
        increasing = all(xl[i] < xl[i + 1] for i in range(len(xl) - 1))

    if not increasing:
        return ["not increasing sequence"] * n_derivs

    cs = CubicSpline(x, y)
    derivs_averages = []
    for n in range(n_derivs):
        n_deriv = cs(x, n)
        # use true division to get a float average
        deriv_average = float(sum(n_deriv)) / len(n_deriv)

        derivs_averages.append(deriv_average)

    return derivs_averages