import numpy as np
import pandas as pd
import time
import librosa

from sklearn.linear_model import SGDClassifier

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

def make_feature_set(spec, spec_dt, annotation_path, w_length):
    """
    Generates features and labels from spectrogram.
    :param spec: NxM numpy array
    :param spec_dt: float
    :param annotation_path: String
    :param w_length: int
    :return: features, labels
    """
    if annotation_path is not None:
        df = pd.read_csv(annotation_path, header=None, 
            names=['onsets', 'offsets', 'label'], delimiter='\t')
        onsets = np.asarray(df['onsets'])
        offsets = np.asarray(df['offsets'])

    num_features = spec.shape[1]-w_length+1

    # Generate features
    features = [[]] * num_features
    for i in xrange(num_features):
    	features[i] = np.ravel(spec[:, i:i+w_length])

    # Generate labels
    if annotation_path is not None:
        labels = np.zeros(num_features)
        for on, off in zip(onsets, offsets):
            center = (on+off)/2
            start = int(np.round((center-0.100)/spec_dt))
            finish = int(np.round((center+0.100)/spec_dt))
            labels[start:finish+1] = 1
        return features, labels

    return features

if __name__ == "__main__":
    filename = "../../audio/SBI-1_20090915_234016.wav"
    annotation_path = "../../annotations/SBI-1_20090915_HAND_LOW_IDaek_EDITED_with_HIGH.txt"
    model_path = "../../features/SBI_coefs_41.npy"

    # Load audio and compute spectrogram
    sr = 24000
    n_fft = 256 # =win_length
    win_length = n_fft
    hop_length = 128.0
    hop_size = hop_length/sr # in seconds
    win_size = win_length/sr # in seconds
    spec_dt = hop_length/sr # in seconds

    # Define length of coefficient window
    w_length = 5

    # Iterate through signal by large blocks (constrained by RAM)
    num_hops_per_block = 10000
    block_i = 0
    done = False
    duration = 1000          #seconds
    block_size = hop_size*num_hops_per_block
    # duration = num_hops_per_block*hop_size+win_size     # in secs

    ####################################################
    # Train model
    model = SGDClassifier(loss='log')

    while not done:
        print "Processing next block... i={} ".format(block_i) + str(time.clock())
        offset = block_i*block_len
        y, _ = librosa.load(filename, offset=offset, duration=block_size, sr=sr)
        if len(y) < block_size*sr:
            print "last one!"
            # BAD!!! Throwing out last bit of data
            break
            done = True
        D = np.log(np.abs(librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length)))

        (features, labels) = make_feature_set(D, spec_dt, annotation_path,
            	w_length)

        actual_features = np.square(half_rectify(np.absolute(features[:, 1:]) -
                                         np.absolute(features[:, :-1]))), 0)

        model.partial_fit(actual_features,labels,classes=[0,1])

        block_i += 1

    coefs = model.coef_.reshape((n_fft/2 + 1, w_length))

    np.save(model_path, coefs)

    ####################################################
    # Evaluate model
    test_path = '../../audio/NSDNS_20110902_192900.wav'
    detection_curve_path = '../../detection_functions/NSDNS_SF_42.npy'

    coefs = np.load(model_path)
    c = coefs.reshape(645,1)
    block_i = 0
    duration = 1000          #seconds
    done = False
    detection_curve = np.zeros(0)

    block_len = num_hops_per_block-w_length+2

    while not done:
        print "Testing next block... i={} ".format(block_i) + str(time.clock())
        offset = block_i*block_len
        y, _ = librosa.load(test_path, offset=offset, duration=block_size, sr=sr)
        if len(y) < block_size*sr or (duration is not None and offset > duration):
            print "last one!"
            # BAD!!! Throwing out last bit of data
            break
            done = True
        D = np.log(np.abs(librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length)))

        features = np.asarray(make_feature_set(D, spec_dt, annotation_path=None, 
            w_length=w_length))

        features = features.T

        print features.shape

        # Template detector
        # detection_curve = np.concatenate((detection_curve,features.dot(c)), axis=1)

        # Multi-frame SF
        detection_curve = np.concatenate((detection_curve,spectral_flux(features*c)), axis=1)

        block_i += 1

    np.save(detection_curve_path, detection_curve)

    print spec_dt



# for b,y in sample(x):        # I write sample function. b is the size of the batch.
#     model.partial_fit(b,y)   # y is label. for SGD use SVM or logistic regression w/ L2
