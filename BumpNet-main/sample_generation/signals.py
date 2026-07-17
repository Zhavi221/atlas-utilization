from scipy import stats

def gaussian(x, mean, width):
    """
    Creates a normal (area 1) Gaussian with a given mean and width
    mean - bin number
    width - number of bins
    """
    return stats.norm.pdf(x, loc=mean, scale=width)
