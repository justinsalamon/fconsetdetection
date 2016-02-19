import numpy as np
import mir_eval
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve

import scipy.signal as sg


def plot_pr_curve(recalls, precisions, save=False, outfile=None):
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
    plt.grid()

    if not save:
        plt.show()
    else:
        plt.savefig(outfile[:-3] + '.png')


def eval_detection_func(annotation_path, function_path, start_time, dt, trial_number,
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

    ref_offsets = np.asarray(df['offsets'])
    ref_offsets = ref_offsets[ref_offsets >= start_time]
    ref_offsets = ref_offsets[ref_offsets < t_end]

    assert len(ref_onsets) == len(ref_offsets)

    mir_eval.onset.MAX_TIME = t_end

    #### Pre-process detection function ####

    # Remove nans just in case
    detection_function = detection_function[~np.isnan(detection_function)]

    # Normalize
    detection_function -= np.mean(detection_function)
    detection_function /= np.max(np.abs(detection_function))

    # Smooth
    win_size = 29
    # max_filt_len = int(0.200/dt)
    # print max_filt_len
    b, a = sg.butter(1, 6*dt)   # multiplying by dt = dividing by fs
    detection_function_smoothed = sg.filtfilt(b, a, detection_function)
    detection_function = sg.medfilt(detection_function_smoothed, win_size)
    # detection_function = sg.medfilt(detection_function, win_size)

    # detection_function = maxfilt(detection_function_smoothed, win_size)
    # detection_function = maxfilt(detection_function, win_size)

    # Generate list of every peak and corresponding value
    peak_mask = make_peak_mask(detection_function, 0)
    y_pred = peak_mask*detection_function

    # Generate corresponding binary thing
    y_true = np.zeros(y_pred.shape)
    for on, off in zip(ref_onsets, ref_offsets):
        center = (on+off)/2
        start = int(np.round((center-0.100)/dt))
        finish = int(np.round((center+0.100)/dt))
        y_true[start:finish+1] = 1
    y_true *= peak_mask

    indices = y_pred > 0
    y_pred = y_pred[indices]
    y_true = y_true[indices]

    P, R, T = precision_recall_curve(y_true, y_pred)
    T = np.append(T, 0)
    eps = 0.000000001
    F = 2*P*R/(P+R+eps)
    F2 = 5*P*R/(4*P+R+eps)

    print max(F)
    print max(F2)

    pr = np.column_stack((P, R, T, F, F2))
    outfile = '../../results/' + function_path[26:-4] + '_' + str(trial_number) + '_new_way.txt'
    np.savetxt(outfile, pr)

    plot_pr_curve(R, P, save=True, outfile=outfile)


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


def make_peak_mask(detection_function, threshold):
    """
    Given the output of a detection function, finds the indices of the peaks
    :param detection_function: numpy array
    :param threshold: float
    :return: detection function where all indices except for peaks are 0
    """

    smaller = detection_function[0:-2] <= detection_function[1:-1]
    larger = detection_function[1:-1] > detection_function[2:]
    over_threshold = detection_function[1:-1] >= threshold
    peaks = smaller*larger*over_threshold
    mask = np.ones(detection_function.shape)
    mask[1:-1] *= peaks
    return mask


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


def maxfilt(detection_function, win_size):
    out = np.zeros_like(detection_function)
    offset_l = np.floor(win_size/2)
    offset_r = win_size-offset_l-1
    for i in xrange(win_size):
        out[offset_l:-1*offset_r] = np.maximum(detection_function[offset_l:-1*offset_r],
                                                   detection_function[i:i+len(out)-win_size+1])
    return out


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

# path_ref = "../../annotations/SBI-1_20090915_HAND_LOW_IDaek_EDITED_with_HIGH.txt"
# path_est = "../../detection_functions/SBI-1_20090915_234016_SF_36.npy"
# start_time = 0
# dt = 0.00533333333333

# path_ref = "../../annotations/ALFRED_20110924_183200.HAND_high_442NFCs_IDaek_EDIT_TO_INCLUDE_ALL.txt"
# path_est = "../../detection_functions/ALFRED_20110924_183200_0-3600_SVM_8.npy"

# path_ref = "../../annotations/SBI-1_20090915_HAND_LOW_IDaek_EDITED_with_HIGH.txt"
# path_est = "../../detection_functions/SBI-1_20090915_234016_KNN_12.npy"
# start_time = 0

# path_ref = "../../annotations/NSDNS_20110902_192900_high_and_low.txt"
# path_est = "../../detection_functions/NSDNS_toy3.npy"
# dt = 0.05       # Time between every prediction

path_ref = "../../annotations/NSDNS_20110902_192900_high_and_low.txt"
path_est = "../../detection_functions/NSDNS_20110902_192900_SF_14.npy"
dt = 128.0/24000      # Time between every prediction
start_time = 0
#
eval_detection_func(path_ref, path_est, start_time, dt, trial_number=670, duration=None)


####### Following code for operating with Cornell data
# path_ref_pre= "../../annotations/"
# path_prefix = "../../detection_functions/NFC_correlation_raw_data/"

# path_ests = ["amre_corr_confnorm.npy", "chsp_corr_confnorm.npy", "oven_corr_confnorm.npy", "savs_corr_confnorm.npy",
#              "sosp_corr_confnorm.npy", "veer_corr_confnorm.npy", "woth_corr_confnorm.npy", "wtsp_corr_confnorm.npy"]
# path_refs = ["nsdns_AMRE.txt", "nsdns_CHSP.txt", "nsdns_OVEN.txt", "nsdns_SAVS.txt",
#              "nsdns_SOSP.txt", "nsdns_VEER.txt", "nsdns_WOTH.txt", "nsdns_WTSP.txt"]
# dts = [0.0050793650679580062, 0.00507936509305, 0.0050793650634, 0.00507936503602,
#        0.0101587300218, 0.0116099772364, 0.0101587302454, 0.00544217690008]

# path_ests = ["amre_corr_conf_resample.npy", "chsp_corr_conf_resample.npy",
#              "oven_corr_conf_resample.npy", "savs_corr_conf_resample.npy",
#              "sosp_corr_conf_resample.npy", "veer_corr_conf_resample.npy",
#              "woth_corr_conf_resample.npy", "wtsp_corr_conf_resample.npy"]
# path_refs = ["nsdns_AMRE.txt", "nsdns_CHSP.txt", "nsdns_OVEN.txt", "nsdns_SAVS.txt",
#              "nsdns_SOSP.txt", "nsdns_VEER.txt", "nsdns_WOTH.txt", "nsdns_WTSP.txt"]
# dts = [0.05, 0.05, 0.05, 0.05,
#        0.05, 0.05, 0.05, 0.05]

# path_ests = ["amre_corr_conf_filt.npy", "chsp_corr_conf_filt.npy", "oven_corr_conf_filt.npy", "savs_corr_conf_filt.npy",
#              "sosp_corr_conf_filt.npy", "veer_corr_conf_filt.npy", "woth_corr_conf_filt.npy", "wtsp_corr_conf_filt.npy"]
# dts = [0.050793650679580062, 0.0507936509305, 0.050793650634, 0.0507936503602,
#        0.507936501090, 0.0116099772364*5, 0.0101587302454*5, 0.0544217690008]
#
# path_ests = ["nsdns_resampled_all_max.npy"]
# dts = [0.05]
# dts = [0.050793650679580062]

# path_ests = ["nsdns_resample2_all_max.npy"]
# path_refs = ["NSDNS_20110902_192900_high_and_low.txt"]
# dts = [0.005079365036]

# path_ests = ["random_noise.npy"]
# path_refs = ["NSDNS_20110902_192900_high_and_low.txt"]
# dts = [0.05]

#
# start_times = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#
# # i = 0
# # #
#
# trial_number = 28
# for i in xrange(8):
#     path_est = path_ests[i]
#     dt = dts[i]
#     start_time = start_times[i]
#     eval_detection_func(path_ref_pre+path_refs[i], path_prefix+path_est, start_time, dt, trial_number+i, duration=None)

