from scipy.interpolate import CubicSpline

def calc_n_derivs_avg(x, y, n_derivs):
    cs = CubicSpline(x, y)
    derivs_averages = []
    for n in range(n_derivs):
        n_deriv = cs(x, n)
        deriv_average = sum(n_deriv)//len(n_deriv)
        
        derivs_averages.append(deriv_average)

    return derivs_averages

#FEATURE 
#add interpolation for statistics and calc first and second deriviate in order to show progress over time

# https://docs.scipy.org/doc/scipy/tutorial/interpolate.html
# https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html


#FEATURE to add more metrics regarding trends in parsing
#memory trends, chunks sizes trends etc