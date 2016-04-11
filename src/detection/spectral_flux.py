import numpy as np
import librosa
import time

def spectral_flux(spec):
    """
    Computes the spectral flux of a spectrogram
    :param spec: a 2-D array of floats
    :return: a numpy array
    """
    return np.sum(np.square(half_rectify(np.absolute(spec[:, 1:]) -
                                         np.absolute(spec[:, :-1]))), 0)


def half_rectify(n):
    return np.fmax(n, np.zeros_like(n))

if __name__ == "__main__":
    filename = '../../audio/NSDNS_20110902_192900.wav'
    outfile = "../../detection_functions/NSDNS_SF_44.npy"

    # Load audio and compute spectrogram
    sr = 24000
    n_fft = 256         # =win_length
    win_length = n_fft
    hop_length = 128.0

    hop_size = hop_length/sr # in seconds
    win_size = win_length/sr # in seconds
    dt = hop_length/sr

    block_size = hop_size*num_hops_per_block+win_size # in secs

    num_hops_per_block = 10000

    streaming_prob = np.asarray([])

    duration = 1000

    block_i = 0
    # Iterate through signal by large blocks (constrained by RAM)
    done = False
    while not done:
        print "Predicting next block..." + str(time.clock())
        offset = block_i*block_size
        y, _ = librosa.load(filename, offset=offset, duration=block_size, sr=sr)
        if len(y) < block_size*sr or (duration is not None and offset > duration):
            print "last one!"
            # BAD!!! Throwing out last bit of data
            break
            done = True
        D = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length)
        print D.shape

        streaming_prob = np.append(streaming_prob, spectral_flux(D))
        block_i += 1

    streaming_prob /= np.max(streaming_prob)
    inv_streaming_prob = np.ones_like(streaming_prob)-streaming_prob
    out = np.column_stack((streaming_prob, inv_streaming_prob))

    np.save(outfile, out)
    print dt

