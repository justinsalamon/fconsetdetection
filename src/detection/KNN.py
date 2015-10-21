import numpy as np
import librosa
import os
import sklearn.neighbors


def train_KNN()

def KNN(spec):
    """
    Computes the spectral flux of a spectrogram
    :param spec: a 2-D array of floats
    :return: a numpy array
    """

    # Make features
    for

    # Test




def half_rectify(n):
    return np.fmax(n, np.zeros_like(n))

if __name__ == "__main__":
    filename = "../../audio/ALFRED_20110924_183200.wav"
    outfile = "../../detection_functions/ALFRED_20110924_183200_SF.npy"

    # Load audio and compute spectrogram
    y, sr = librosa.core.load(filename, sr=None)
    n_fft = 256         # =win_length
    hop_length = 128.0
    dt = hop_length/sr
    D = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length)

    streaming_prob = spectral_flux(D)
    streaming_prob /= np.max(streaming_prob)
    inv_streaming_prob = np.ones_like(streaming_prob)-streaming_prob
    out = np.column_stack((streaming_prob, inv_streaming_prob))

    np.save(outfile, out)
    print dt

