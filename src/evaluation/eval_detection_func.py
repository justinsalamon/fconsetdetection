import numpy as np
import mir_eval
import matplotlib.pyplot as plt
import pandas as pd


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
                        time_limit=3000):
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

    # Set testing params
    t_end = start_time + time_limit    # seconds

    # Detection function output to test
    detection_function = np.load(function_path)

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

    mir_eval.onset.MAX_TIME = t_end

    # smooth detection function
    win_size = 2        # length of mean filter - 2 or 3 seems to work best
    detection_function = mean_filter(detection_function, win_size)

    for threshold in np.linspace(0, 1, 100):
        est_onsets_ind = pick_peaks_with_smoothing(detection_function, threshold, win_size)
        # est_onsets_ind = pick_peaks(detection_function, threshold)
        est_onsets = indices_to_times(est_onsets_ind, start_time, dt)
        F, P, R = mir_eval.onset.f_measure(ref_onsets,
                                           est_onsets,
                                           window=0.2)
        out.append((threshold, (F, P, R)))
        precisions.append(P)
        recalls.append(R)
        max_F = max(max_F, F)

    print max_F

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
    locs = np.where(peaks == True)[0] + win_size - 1
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

path_ref = "../../annotations/ALFRED_20110924_183200.HAND_high_442NFCs_IDaek_EDIT_TO_INCLUDE_ALL.txt"
path_est = "../../detection_functions/test.npy"
start_time = 500
dt = 0.005

eval_detection_func(path_ref, path_est, start_time, dt, time_limit=100)
