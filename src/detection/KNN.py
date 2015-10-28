import numpy as np
import librosa
import os
import sklearn.neighbors as nbrs
import scipy.signal as sg


def compute_features(y, sr):
    """
    Given a short sample of audio, output the features we are training for
    :param y: 1xN array
        audio data
    :param sr: int
        sampling rate
    :return: feature vector
    """
    mfccs = librosa.feature.mfcc(y, sr, n_mfcc=50, n_fft=2048, hop_length=220,
                                    n_mels=40, fmin=2000)

    mfccs = mfccs[1:,:]

    # Summarize features
    # print np.shape(mfccs)
    features = np.mean(mfccs, axis=1)
    features = np.append(features, np.std(mfccs, axis=1))
    d_mfccs = np.diff(mfccs, axis=1)
    features = np.append(features, np.mean(d_mfccs, axis=1))
    features = np.append(features, np.std(d_mfccs, axis=1))
    d_d_mfccs = np.diff(d_mfccs, axis=1)
    features = np.append(features, np.mean(d_d_mfccs, axis=1))
    features = np.append(features, np.std(d_d_mfccs, axis=1))

    # print np.shape(d_d_mfccs)
    #
    # print np.shape(features)
    return np.reshape(features, (1, len(features)))


def train_KNN(clf, sample_dir):
    """
    Trains a classifier on all contents of a given directory. Labels are given
        in the name of the file
    :param clf: sklearn classifier
    :param sample_dir: String
    :return: sklearn classifier
    """

    # create data and labels
    data = None
    labels = None

    # fill data and labels
    for f in os.listdir(sample_dir):
        if f[-4:] != '.wav':
            continue
        y, sr = librosa.load(sample_dir + f, sr=None)
        features = compute_features(y, sr)

        # THERE EXISTS A BETTER WAY OF HANDLING THIS WHOLE THING
        if data is None:
            data = features
        else:
            data = np.concatenate((data, features), axis=0)

        if f[0] == 't':
            if labels is None:
                labels = np.asarray([1])
            else:
                labels = np.concatenate((labels, [1]))
        elif f[0] == 'f':
            if labels is None:
                labels = np.asarray([0])
            else:
                labels = np.concatenate((labels, [0]))
        else:
            print "BAD FILENAMES!"
            exit()

    # print labels
    # print np.shape(data)

    # Fit classifier
    clf.fit(data, labels)

    return clf


def predict_KNN(clf, y, sr, win_size=0.15, hop_size=0.05):
    """
    "Onset detection function." Outputs a running classification of 150ms
    windows.
    :param clf: sklearn classifier
    :param y: 1xN array
        full audio data
    :param dt: float
        time between frames in seconds
    :param win_size: float
        window size in seconds
    :param hop_size: float
        hop size in seconds
    :return: (1xL, float)
        novelty curve, time between samples on novelty curve in seconds
    """

    win_samples = np.floor(win_size*sr)   # number of samples in window
    hop_samples = np.floor(hop_size*sr)   # number of samples in hop

    novelty_curve = np.asarray([])

    i = 0
    while i+win_samples < len(y):
        segment = y[i:i+hop_samples]
        features = compute_features(segment, sr)
        label = clf.predict_proba(features)[0][1]
        # print label
        novelty_curve = np.append(novelty_curve, label)
        i += hop_samples

    # print (novelty_curve[0:50])

    return novelty_curve, hop_size


def half_rectify(n):
    return np.fmax(n, np.zeros_like(n))

if __name__ == "__main__":
    sample_dir = "../../audio/samples/ALFRED/"
    infile = "../../audio/ALFRED_20110924_183200_0-3600.wav"
    outfile = "../../detection_functions/ALFRED_20110924_183200_0-3600_KNN_8.npy"

    # Train classifier
    print "Training classifier..."
    clf = nbrs.KNeighborsClassifier(n_neighbors=5)
    clf = train_KNN(clf, sample_dir)

    # Run classifier on 150ms windows of data
    print "Loading audio..."
    y, sr = librosa.load(infile, sr=None)
    print "Predicting novelty curve..."
    onsets, dt = predict_KNN(clf, y, sr)

    # Smooth so taking median later doesn't kill everything
    streaming_prob = sg.convolve(onsets, [0.1, 0.8, 0.1])
    inv_streaming_prob = np.ones_like(streaming_prob)-streaming_prob
    out = np.column_stack((streaming_prob, inv_streaming_prob))

    np.save(outfile, out)
    print dt

