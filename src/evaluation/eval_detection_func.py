import numpy as np
import mir_eval
import matplotlib.pyplot as plt
import pandas as pd

import scipy.signal as sg


def plot_pr_curve(recalls, precisions):
    '''
    Given array of recall values and array of precision values, plot PR curve
    :param recalls:
    :param precisions:
    :return:
    '''
    assert len(recalls) == len(precisions)

    # Create graph
    plt.clf()
    plt.plot(recalls, precisions, '-o', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve: 200ms Tolerance')
    plt.legend(loc="lower left")
    plt.show()


def eval_detection_func(annotation_path, function_path, start_time, dt,
                        duration=3000):
    """
    Evaluates a detection function. Given the output of a detection function we
    find the peaks, and compare them to a list of ground-truth annotations.
    Then print out the maximum F-measure found and display a precision-recall
    graph.
    :param annotation_path: String
    :param function_path: String
    :param start_time: float
    :param dt: float
    :return: ???
    """

    # Detection function output to test
    detection_function = np.load(function_path)

    # Set testing params
    if duration is None:
        t_end = len(detection_function) * dt
    else:
        t_end = start_time + duration    # seconds

    # Downmix
    print np.shape(detection_function)
    if len(np.shape(detection_function)) > 1:
        detection_function = detection_function[:, 0]

    # plt.plot(detection_function)


    # est_times = indices_to_times(np.arange(len(detection_function)),
    #                              start_time, dt)
    # limit_ind = np.where(est_times < time_limit)[0][-1]
    # est_times = est_times[:limit_ind+1]
    # detection_function = detection_function[:limit_ind+1]

    # If detection function times start at 0 instead of start_time, add offset
    # This is totally not foolproof, but will work most of the time
    if detection_function[0] < start_time:
        detection_function += start_time

    # Get reference onsets for ground truth
    df = pd.read_csv(annotation_path, header=None,
                     names=['onsets', 'offsets', 'label'], delimiter='\t')
    ref_onsets = np.asarray(df['onsets'])
    ref_onsets = ref_onsets[ref_onsets >= start_time]
    ref_onsets = ref_onsets[ref_onsets < t_end]

    # Get list of precisions and recalls for varying thresholds
    out = []
    precisions = []
    recalls = []
    max_F = 0
    max_F2 = 0

    mir_eval.onset.MAX_TIME = t_end

    # Pre-process detection function

    # Normalize
    detection_function -= np.mean(detection_function)
    detection_function /= np.max(np.abs(detection_function))

    # Smooth
    win_size = 3
    b, a = sg.butter(1, 6*dt)   # multiplying by dt = dividing by fs
    detection_function_smoothed = sg.filtfilt(b, a, detection_function)
    detection_function_med = sg.medfilt(detection_function_smoothed, win_size)

    eps = 0.000000001

    for threshold in np.linspace(0, 1, 100):
        # est_onsets_ind = pick_peaks_with_smoothing(detection_function, threshold, win_size)
        a_threshold = threshold + detection_function_med
        est_onsets_ind = pick_peaks_at(detection_function, a_threshold)
        # est_onsets_ind = pick_peaks(detection_function, threshold)
        est_onsets = indices_to_times(est_onsets_ind, start_time, dt)
        F, P, R = mir_eval.onset.f_measure(ref_onsets,
                                           est_onsets,
                                           window=0.2)
        F2 = 5*P*R/(4*P+R+eps)
        out.append((threshold, (F, P, R)))
        precisions.append(P)
        recalls.append(R)
        max_F = max(max_F, F)
        max_F2 = max(max_F2, F2)

    print max_F
    print max_F2

    plot_pr_curve(recalls, precisions)


def pick_peaks(detection_function, threshold):
    """
    Given the output of a detection function, finds the indices of the peaks
    :param detection_function: numpy array
    :param threshold: float
    :return: numpy array of ints
    """

    smaller = detection_function[0:-2] <= detection_function[1:-1]
    larger = detection_function[1:-1] > detection_function[2:]
    over_threshold = detection_function[1:-1] >= threshold
    peaks = smaller*larger*over_threshold
    locs = np.where(peaks == True)[0] + 1
    return locs


def pick_peaks_at(detection_function, a_threshold):
    """
    Given the output of a detection function and an adaptive threshold, finds
     the indices of the peaks
    :param detection_function: 1xN vector
    :param threshold: 1xN vector
    :return: numpy array of ints
    """

    smaller = detection_function[0:-2] <= detection_function[1:-1]
    larger = detection_function[1:-1] > detection_function[2:]
    above_thresh = detection_function > a_threshold
    peaks = smaller*larger*above_thresh[1:-1]
    locs = np.where(peaks == True)[0] + 1
    return locs


def mean_filter(detection_function, win_size):
    # Normalize
    detection_function -= np.mean(detection_function)
    detection_function /= np.max(np.abs(detection_function))

    # Smooth with variable length mean filter
    smoothed_function = np.zeros(detection_function.size-win_size+1)
    for i in xrange(win_size-1):
        smoothed_function += detection_function[i: -1*win_size+i+1]
    smoothed_function += detection_function[win_size-1:]
    return smoothed_function/win_size


def pick_peaks_with_smoothing(detection_function, threshold, win_size):
    """
    Same as pick_peaks, but incorporates a simple low pass filter
    :param detection_function: numpy array
    :param threshold: float
    :param win_size: int
        the offset due to frames dropped while filtering
    :return: numpy array of ints
    """

    smaller = detection_function[0:-2] <= detection_function[1:-1]
    larger = detection_function[1:-1] > detection_function[2:]
    over_threshold = detection_function[1:-1] >= threshold

    peaks = smaller*larger*over_threshold
    # locs = np.where(peaks == True)[0] + win_size - 1
    locs = np.where(peaks == True)[0] + 1
    return locs


def times_to_indices(times, start_time, dt):
    """
    Converts a time or series of times to their corresponding indices in the
    detection function
    :param times: float or collection of floats
    :param start_time: float
    :param dt: float
    :return: int or numpy array of int
    """
    return int(np.round((times - start_time)/dt))


def indices_to_times(indices, start_time, dt):
    """
    Converts a series of detection function indices to their corresponding
    times
    :param indices: int or collection of ints
    :param start_time: float
    :param dt: float
    :return: float or numpy array of floats
    """
    return start_time + indices * dt


# path_ref = "../../annotations/NSDNS_20110902_192900_high_and_low.txt"
# path_est = "../../detection_functions/NSDNS_20110902_192900_streaming_prob.npy"
# dt = 0.05079365079
# start_time = 0.07619047619

# path_ref = "../../annotations/ALFRED_20110924_183200.HAND_high_442NFCs_IDaek_EDIT_TO_INCLUDE_ALL.txt"
# path_est = "../../detection_functions/ALFRED_20110924_183200_0-3600_SF.npy"
# start_time = 0
# dt = 0.00533333333333

# path_ref = "../../annotations/ALFRED_20110924_183200.HAND_high_442NFCs_IDaek_EDIT_TO_INCLUDE_ALL.txt"
# path_est = "../../detection_functions/ALFRED_20110924_183200_0-3600_SVM_8.npy"

# path_ref = "../../annotations/SBI-1_20090915_HAND_LOW_IDaek_EDITED_with_HIGH.txt"
# path_est = "../../detection_functions/SBI-1_20090915_234016_KNN_12.npy"
# start_time = 0
# dt = 0.05       # Time between every prediction
#
# eval_detection_func(path_ref, path_est, start_time, dt, duration=None)


######## Following code for operating with Cornell data
path_prefix = "../../detection_functions/NFC_correlation_raw_data/"
path_ests = ["amre_corr_conf.npy", "chsp_corr_conf.npy", "oven_corr_conf.npy", "savs_corr_conf.npy",
             "sosp_corr_conf.npy", "veer_corr_conf.npy", "woth_corr_conf.npy", "wtsp_corr_conf.npy"]
path_ref = "../../annotations/NSDNS_20110902_192900_high_and_low.txt"
dts = [0.0050793650679580062, 0.00507936509305, 0.0050793650634, 0.00507936503602,
       0.0101587300218, 0.0116099772364, 0.0101587302454, 0.00544217690008]
start_times = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

i = 5

path_est = path_ests[i]
dt = dts[i]
start_time = start_times[i]
eval_detection_func(path_ref, path_prefix+path_est, start_time, dt, duration=None)

