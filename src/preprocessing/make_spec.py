import librosa
import numpy as np

# MAYBE THIS FILE IS UNNECESSARY


def make_stft(filepath, n_fft=256, hop_length=None, win_length=None, window=None):
    """
    Creates and saves an STFT from an audio file
    :param filepath: string
    :param n_fft: int
    :param hop_length: int
    :param win_length: int
    :param window: None, vector, or function
    :return: numpy array of complex numbers
    """

    # load audio
    y, sr = librosa.core.load(filepath, sr=None)

    # get stft
    D = librosa.core.stft(y, n_fft, hop_length, win_length, window)

    # rename
    i = filepath.rfind(".")
    out_filepath = filepath[:i] + ".stft"   # INCORRECT - should save in different dir.

    np.save(out_filepath, D)

