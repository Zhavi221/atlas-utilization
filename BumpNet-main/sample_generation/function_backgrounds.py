#-----------------------------------------------------------------------------#
# Module for generating smoothly decaying background distributions.
#
# Currently implements:
#   - exponential
#   - linear
#-----------------------------------------------------------------------------#
import numpy as np

def exponential_params(x1, y1, x2, y2):
    """
    Return parameters of exponential decay function between two points
    (x1, y1) and (x2, y2), where x1 < x2 and y1 > y2.

    Function : y(x) = p0 * exp(-p1 * x)

    Parameters
    ----------
    x1, y1, x2, y2 : float
        xy-coordinates of lower (x1, y1) and upper (x2, y2) end points

    Returns
    -------
    p0, p1 : float
        function parameters
    """
    p1 = (np.log(y1/y2))/(x2 - x1)
    p0 = y1*np.exp(p1*x1)
    return p0, p1

def exponential(x, p0, p1):
    """
    Evaluate at x an exponential decay function parametrized by p0, p1.

    Function : y(x) = p0 * exp(-p1 * x)

    Parameters
    ----------
    x : array_like
        x-values to evaluate the function at
    p0, p1 : float
        function parameters

    Returns
    -------
    y : array_like
        evaluated function values
    """
    return p0 * np.exp(-p1 * x)

def linear_params(x1, y1, x2, y2):
    """
    Return parameters of linear decay function between two points
    (x1, y1) and (x2, y2), where x1 < x2 and y1 > y2.

    Function : y(x) = p0 * x + p1

    Parameters
    ----------
    x1, y1, x2, y2 : float
        xy-coordinates of lower (x1, y1) and upper (x2, y2) end points

    Returns
    -------
    p0, p1 : float
        function parameters
    """
    p0 = (y2 - y1)/(x2 - x1)
    p1 = (x2*y1 - x1*y2)/(x2 - x1)
    return p0, p1

def linear(x, p0, p1):
    """
    Evaluate at x a linear decay function parametrized by p0, p1.

    Function : y(x) = p0 * x + p1

    Parameters
    ----------
    x : array_like
        x-values to evaluate the function at
    p0, p1 : float
        function parameters

    Returns
    -------
    y : array_like
        evaluated function values
    """
    return p0 * x + p1

def one_over_x_params(x1, y1, x2, y2):
    """
    Return parameters of smoothly decaying function between two points
    (x1, y1) and (x2, y2), where x1 < x2 and y1 > y2.

    Function : y(x) = 1/(p0 * x) + p1

    Parameters
    ----------
    x1, y1, x2, y2 : float
        xy-coordinates of lower (x1, y1) and upper (x2, y2) end points

    Returns
    -------
    p0, p1 : float
        function parameters
    """
    p0 = -(x1 - x2)/(x1*x2*y1 - x1*x2*y2)
    p1 = (x1*y1 - x2*y2)/(x1 - x2)
    return p0, p1

def one_over_x(x, p0, p1):
    """
    Evaluate at x a smoothly decaying function parametrized by p0, p1.

    Function : y(x) = 1/(p0 * x) + p1

    Parameters
    ----------
    x : array_like
        x-values to evaluate the function at
    p0, p1 : float
        function parameters

    Returns
    -------
    y : array_like
        evaluated function values
    """

    return 1/(p0*x) + p1

def one_over_x_squared_params(x1, y1, x2, y2):
    """
    Return parameters of smoothly decaying function between two points
    (x1, y1) and (x2, y2), where x1 < x2 and y1 > y2.

    Function : y(x) = 1/(p0 * x**2) + p1

    Parameters
    ----------
    x1, y1, x2, y2 : float
        xy-coordinates of lower (x1, y1) and upper (x2, y2) end points

    Returns
    -------
    p0, p1 : float
        function parameters
    """
    p0 = (x2**2 - x1**2)/(x1**2*x2**2*(y1 - y2))
    p1 = (x1**2*y1 - x2**2*y2)/(x1**2 - x2**2)
    return p0, p1

def one_over_x_squared(x, p0, p1):
    """
    Evaluate at x a smoothly decaying function parametrized by p0, p1.

    Function : y(x) = 1/(p0 * x**2) + p1

    Parameters
    ----------
    x : array_like
        x-values to evaluate the function at
    p0, p1 : float
        function parameters

    Returns
    -------
    y : array_like
        evaluated function values
    """

    return 1/(p0*x**2) + p1

def one_over_x_cubed_params(x1, y1, x2, y2):
    """
    Return parameters of smoothly decaying function between two points
    (x1, y1) and (x2, y2), where x1 < x2 and y1 > y2.

    Function : y(x) = 1/(p0 * x**2) + p1

    Parameters
    ----------
    x1, y1, x2, y2 : float
        xy-coordinates of lower (x1, y1) and upper (x2, y2) end points

    Returns
    -------
    p0, p1 : float
        function parameters
    """
    p0 = (x2**3 - x1**3)/(x1**3*x2**3*(y1 - y2))
    p1 = (x1**3*y1 - x2**3*y2)/(x1**3 - x2**3)
    return p0, p1

def one_over_x_cubed(x, p0, p1):
    """
    Evaluate at x a smoothly decaying function parametrized by p0, p1.

    Function : y(x) = 1/(p0 * x**3) + p1

    Parameters
    ----------
    x : array_like
        x-values to evaluate the function at
    p0, p1 : float
        function parameters

    Returns
    -------
    y : array_like
        evaluated function values
    """

    return 1/(p0*x**3) + p1

def one_over_x_to_4th_params(x1, y1, x2, y2):
    """
    Return parameters of smoothly decaying function between two points
    (x1, y1) and (x2, y2), where x1 < x2 and y1 > y2.

    Function : y(x) = 1/(p0 * x**4) + p1

    Parameters
    ----------
    x1, y1, x2, y2 : float
        xy-coordinates of lower (x1, y1) and upper (x2, y2) end points

    Returns
    -------
    p0, p1 : float
        function parameters
    """
    p0 = (x2**4 - x1**4)/(x1**4*x2**4*(y1 - y2))
    p1 = (x1**4*y1 - x2**4*y2)/(x1**4 - x2**4)
    return p0, p1

def one_over_x_to_4th(x, p0, p1):
    """
    Evaluate at x a smoothly decaying function parametrized by p0, p1.

    Function : y(x) = 1/(p0 * x**4) + p1

    Parameters
    ----------
    x : array_like
        x-values to evaluate the function at
    p0, p1 : float
        function parameters

    Returns
    -------
    y : array_like
        evaluated function values
    """
    return 1/(p0*x**4) + p1

def one_over_x_to_nth_params(x1, y1, x2, y2, n):
    """
    Return parameters of smoothly decaying function between two points
    (x1, y1) and (x2, y2), where x1 < x2 and y1 > y2.

    Function : y(x) = cosh(p0*(x - x2)) + p1

    Parameters
    ----------
    x1, y1, x2, y2 : float
        xy-coordinates of lower (x1, y1) and upper (x2, y2) end points

    Returns
    -------
    p0, p1, n: float
        function parameters
    """

    p0 = (x2**n - x1**n)/(x1**n*x2**n*(y1 - y2))
    p1 = (x1**n*y1 - x2**n*y2)/(x1**n - x2**n)

    return  p0, p1

def one_over_x_to_nth(x, p0, p1, n):
    """
    Evaluate at x a smoothly decaying function parametrized by p0, p1 and n.

    Function : y(x) = 1/(p0 * x**4) + p1

    Parameters
    ----------
    x : array_like
        x-values to evaluate the function at
    p0, p1, n : float
        function parameters

    Returns
    -------
    y : array_like
        evaluated function values
    """

    return  1/(p0*x**n) + p1

def parabola_half_params(x1, y1, x2, y2):
    """
    Return parameters of smoothly decaying function between two points
    (x1, y1) and (x2, y2), where x1 < x2 and y1 > y2.

    Function : y(x) = -p0 * (x - x2)**2 + y2

    Parameters
    ----------
    x1, y1, x2, y2 : float
        xy-coordinates of lower (x1, y1) and upper (x2, y2) end points

    Returns
    -------
    p0, x2, y2 : float

    (x2,y2) is the vertex
    x2 is the x value of the minimum vertex
    y2 is the y value of the minimum vertex
        function parameters
    """

    p0 = (y1 - y2)/(x1 - x2)**2
    if isinstance(p0, np.ndarray):
        p1 = np.zeros(p0.shape) #not used for this function
    else:
        p1 = 0
    return p0, p1


#need to add how to properly fix x2,y2
def parabola_half(x, p0, p1, x2, y2):
    """
    Evaluate at x a smoothly decaying function parametrized by p0, p1.

    Function : y(x) = -p0 * (x - x2)**2 + y2

    Parameters
    ----------
    x : array_like
        x-values to evaluate the function at
    p0 : float
        function parameters

    Returns
    -------
    y : array_like
        evaluated function values
    """
    
    if isinstance(p0, np.ndarray):
        y2 = y2[:, None]

    return p0*(x - x2)**2 + y2

def ln_negative_params(x1, y1, x2, y2):
    """
    Return parameters of smoothly decaying function between two points
    (x1, y1) and (x2, y2), where x1 < x2 and y1 > y2.

    Function : y(x) = -p0 * ln(x) + p1

    Parameters
    ----------
    x1, y1, x2, y2 : float
        xy-coordinates of lower (x1, y1) and upper (x2, y2) end points

    Returns
    -------
    p0, p1 : float
        function parameters
    """
    p0 = (y2 - y1)/(np.log(x1) - np.log(x2))
    p1 = (y2*np.log(x1) - y1*np.log(x2))/(np.log(x1) - np.log(x2))
    return p0, p1

def ln_negative(x, p0, p1):
    """
    Evaluate at x a smoothly decaying function parametrized by p0, p1.

    Function : y(x) = -p0 * ln(x) + p1
    Parameters
    ----------
    x : array_like
        x-values to evaluate the function at
    p0, p1 : float
        function parameters

    Returns
    -------
    y : array_like
        evaluated function values
    """

    return -p0*np.log(x) + p1

def cos_quarter_params(x1, y1, x2, y2):
    """
    Return parameters of smoothly decaying function between two points
    (x1, y1) and (x2, y2), where x1 < x2 and y1 > y2.

    Function : y(x) = (y1 - y2) * cos(p0*(x - p1)) + y1 # python notebood

    Parameters
    ----------
    x1, y1, x2, y2 : float
        xy-coordinates of lower (x1, y1) and upper (x2, y2) end points

    Returns
    -------
    p0, p1 : float
        function parameters
    """
    if isinstance(y1, np.ndarray):
        p0, p1 = np.zeros(y1.shape), np.zeros(y1.shape)
        p0[:] = np.pi/(2*(x2-x1))
        p1[:] = 2*x1 - x2
    else:
        p0 = np.pi/(2*(x2-x1))
        p1 = 2*x1 - x2

    return p0, p1

#need to add how to properly fix x1,x2,y1,y2
def cos_quarter(x, p0, p1, y1, y2):
    """
    Evaluate at x a smoothly decaying function parametrized by p0, p1.

    Function : y(x) = (y1 - y2) * cos(p0*(x - p1)) + y1 # python notebood
    Parameters
    ----------
    x : array_like
        x-values to evaluate the function at
    p0, p1 : float
        function parameters

    Returns
    -------
    y : array_like
        evaluated function values
    """
    if isinstance(y1, np.ndarray):
        y1 = y1[:, None]
        y2 = y2[:, None]

    return (y1 - y2)*np.cos(p0*(x - p1)) + y1

def cosh_half_params(x1, y1, x2, y2):
    """
    Return parameters of smoothly decaying function between two points
    (x1, y1) and (x2, y2), where x1 < x2 and y1 > y2.

    Function : y(x) = cosh(p0*(x - x2)) + p1

    Parameters
    ----------
    x1, y1, x2, y2 : float
        xy-coordinates of lower (x1, y1) and upper (x2, y2) end points

    Returns
    -------
    p0, p1 : float
        function parameters
    """
    p0 = np.arccosh(y1 - y2 + 1)/(x1 - x2)
    p1 = y2 - 1
    return p0, p1

def cosh_half(x, p0, p1, x2):
    """
    Evaluate at x a smoothly decaying function parametrized by p0, p1.

    Function : y(x) = cosh(p0*(x - x2)) + p1

    Parameters
    ----------
    x : array_like
        x-values to evaluate the function at
    p0, p1 : float
        function parameters

    Returns
    -------
    y : array_like
        evaluated function values
    """

    return np.cosh(p0*(x - x2)) + p1