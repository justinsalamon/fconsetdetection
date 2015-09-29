import numpy as np


def spectral_flux(spec, T, F):
    """
    Computes the spectral flux of a spectrogram
    :param spec: a 2-D array of floats
    :param T:
    :param F:
    :return: a numpy array
    """
    return np.sum(np.square(half_rectify(np.absolute(spec[:, 1:]) - np.absolute(spec[:, :-1]))), 1)


def half_rectify(n):
    return max(n, 0)
