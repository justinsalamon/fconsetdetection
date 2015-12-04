import numpy as np
import librosa
import os
import sklearn.neighbors as nbrs
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import scipy.signal as sg
import time


def compute_features_mfcc(y, sr):
    """
    Given a short sample of audio, output the features we are training for
    :param y: 1xN array
        audio data
    :param sr: int
        sampling rate
    :return: mfccs
    """
    # Compute MFCCs...
    hop_length = 256
    mfccs = librosa.feature.mfcc(y, sr, n_mfcc=20, n_fft=1024, hop_length=hop_length,
                                 n_mels=40, fmin=2000)
    # ...and drop first row
    mfccs = mfccs[1:, :]

    sr_mfccs = sr*1.0/hop_length
    return mfccs, sr_mfccs


def summarize_features_mfcc(mfccs, v=False):
    """
    Given mfcc matrix, return summary for a window
    :param mfccs: NxM matrix
        mfcc matrix
    :param i_start: int
        index for beginning of window
    :param i_end: int
        index for end of window
    :return: 1xL array
        feature vector
    """

    # Summarize features
    features = np.max(mfccs, axis=1)
    features = np.append(features, np.mean(mfccs, axis=1))
    features = np.append(features, np.std(mfccs, axis=1))
    d_mfccs = np.diff(mfccs, axis=1)
    features = np.append(features, np.mean(d_mfccs, axis=1))
    features = np.append(features, np.std(d_mfccs, axis=1))
    d_d_mfccs = np.diff(d_mfccs, axis=1)
    features = np.append(features, np.mean(d_d_mfccs, axis=1))
    features = np.append(features, np.std(d_d_mfccs, axis=1))

    # print np.shape(d_d_mfccs)
    # print np.shape(features)
    return np.reshape(features, (1, len(features)))


def train_clf(clfs, sample_dirs, sr):
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
    for sample_dir in sample_dirs:
        for f in os.listdir(sample_dir):
            if f[-4:] != '.wav':
                continue
            y, _ = librosa.load(sample_dir + f, sr=sr)

            features = summarize_features_mfcc(compute_features_mfcc(y, sr)[0])

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
    print np.shape(data)

    # Fit classifier
    for clf in clfs:
        clf.fit(data, labels)

    return clfs


def predict_clf(clf, y, sr, win_size=0.15, hop_size=0.05, append_to=None, t_start=0):
    """
    "Onset detection function." Outputs a running classification of 150ms
    windows.
    :param clf: sklearn classifier
    :param y: 1xN array
        full audio data
    :param sr: float
        sampling rate of audio
    :param win_size: float
        window size in seconds
    :param hop_size: float
        hop size in seconds
    :param append_to: String
        filepath to an existing incomplete detection function
    :return: (1xL, float)
        novelty curve, time between samples on novelty curve in seconds
    """

    win_samples = np.floor(win_size*sr)   # number of samples in window
    hop_samples = np.floor(hop_size*sr)   # number of samples in hop

    if append_to is None:
        novelty_curve = np.asarray([])
    else:
        novelty_curve = np.load(append_to)

    print '\tcomputing mfcc features... ' + str(time.clock())
    mfccs = compute_features_mfcc(y, sr)

    print '\tdoing running summary... ' + str(time.clock())
    i = 0
    ii = 0
    i_offset = t_start * sr
    while True:     # :(
        seg = mfccs[:, i:i+win_samples]
        if seg.shape[1] < win_samples:
            break
        features = summarize_features_mfcc(seg, v=True)
        label = clf.predict_proba(features)[0][1]
        novelty_curve = np.append(novelty_curve, label)
        i += hop_samples
        ii += 1
        if ii % 1000 == 0:
            print 'woo'

    # print (novelty_curve[0:50])

    return novelty_curve, hop_size

def predict_clfs(clfs, filename, outfiles, sr, win_size=0.15, hop_size=0.05, t_start=0):
    """
    "Onset detection function." Outputs a running classification of 150ms
    windows.
    :param clf: sklearn classifier
    :param filename:
        location of audio file
    :param win_size: float
        window size in seconds
    :param hop_size: float
        hop size in seconds
    :param append_to: String
        filepath to an existing incomplete detection function
    :return: (1xL, float)
        novelty curve, time between samples on novelty curve in seconds
    """

    y, _ = librosa.load(filename, duration=1.0, sr=sr) # Just doing this to get sample rate

    # win_samples = np.floor(win_size*sr)   # number of samples in window
    # hop_samples = np.floor(hop_size*sr)   # number of samples in hop

    num_hops_per_block = 10000

    novelty_curves = []
    for i in xrange(len(clfs)):
        novelty_curves.append(np.asarray([]))

    print novelty_curves

    i = 0
    # Iterate through signal by large blocks (constrained by RAM)
    done = False
    duration = num_hops_per_block*hop_size+win_size     # in secs
    while not done:
        print "Predicting next block..." + str(time.clock())
        offset = i*hop_size*num_hops_per_block
        y, _ = librosa.load(filename, offset=offset, duration=duration, sr=sr)
        if len(y) < duration*sr:
            print "last one!"
            # BAD!!! Throwing out last bit of data
            break
            done = True
        mfccs, sr_mfccs = compute_features_mfcc(y, sr)
        print sr_mfccs
        # Iterate through block of MFCC by hop
        h = 0
        while True:     # :(
            seg = mfccs[:, h:h+win_size*sr_mfccs]
            # print win_size*sr_mfccs
            # print seg.shape
            if seg.shape[1] < np.floor(win_size*sr_mfccs):
                break
            features = summarize_features_mfcc(seg, v=True)
            # Predict value for each classifier
            for j in xrange(len(clfs)):
                clf = clfs[j]
                label = clf.predict(features)
                novelty_curves[j] = np.append(novelty_curves[j], label)
            h = np.floor((i+1) * hop_size * sr_mfccs)
            # print novelty_curves
        i += 1

    print "Done predicting. Outputting..." + str(time.clock())
    for i in xrange(len(clfs)):
        novelty_curve = novelty_curves[i]
        outfile = outfiles[i]

        # Smooth so taking median later doesn't kill everything
        streaming_prob = sg.convolve(novelty_curve, [0.1, 0.8, 0.1])
        # Conform data to CLO format
        inv_streaming_prob = np.ones_like(streaming_prob)-streaming_prob
        out = np.column_stack((streaming_prob, inv_streaming_prob))

        print "Rad. Saving output to {}".format(outfile) + str(time.clock())
        np.save(outfile, out)


def half_rectify(n):
    return np.fmax(n, np.zeros_like(n))

if __name__ == "__main__":
    sample_dirs = ["../../audio/samples/ALFRED/", "../../audio/samples/DANBY/"]
    infile = "../../audio/SBI-1_20090915_234016.wav"
    outfiles = ["../../detection_functions/SBI-1_20090915_234016_KNN_13.npy",
                "../../detection_functions/SBI-1_20090915_234016_SVM_13.npy",
                "../../detection_functions/SBI-1_20090915_234016_forest_13.npy"]
    features = None
    # features = ["../../features/SBI-1_20090915_234016_MFCC_20_1024_256_40_2000.npy"]
    sr = 24000

    # Train classifiers
    clf_KNN = nbrs.KNeighborsClassifier(n_neighbors=5)
    clf_SVM = SVC()
    clf_forest = RandomForestClassifier()

    clfs = [clf_KNN, clf_SVM, clf_forest]

    # Train all classifiers at once
    print "Training classifiers... " + str(time.clock())
    train_clf(clfs, sample_dirs, sr=sr)

    print "Getting features... " + str(time.clock())

    print "Predicting..." + str(time.clock())
    predict_clfs(clfs, infile, outfiles, sr=sr)