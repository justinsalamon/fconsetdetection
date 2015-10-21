import librosa
import pandas as pd
import numpy as np

def extract_samples(filename, outdir, annotation_path, t_start=0.0, duration=None):
    '''
    Extracts a 150ms sample containing every annotated call in an audio file.
    Also extracts an equal number of 150ms clips containing no calls. Plops 'em
    all in outdir.

    :param filename: String containing path to audio
    :param outdir: String containing directory for audio to be plopped
    :param annotations: String containing path to annotations
    :param t_start: int
        Where to start extracting (in seconds). Default 0.0
    :param duration: int
        How many seconds of audio to operate on. If None, goes til end.
    :return: None
    '''


    # Get duration of audio
    y, sr = librosa.load(filename, sr=None, offset=t_start, duration=duration)
    sample_len = np.floor(0.075 * sr)   # +/- 75ms
    if duration is not None:
        t_end = t_start + duration

    # Get relevant annotations
    df = pd.read_csv(annotation_path, header=None,
                     names=['onsets', 'offsets', 'label'], delimiter='\t')
    ref_onsets = np.asarray(df['onsets'])
    ref_offsets = np.asarray(df['offsets'])

    i_start = np.searchsorted(ref_onsets, t_start)
    if duration is not None:
        i_end = np.searchsorted(ref_offsets, t_end)

    ref_onsets = ref_onsets[i_start:i_end]
    ref_offsets = ref_offsets[i_start:i_end]

    n_samples = np.size(ref_onsets)

    # Iterate through ref_onsets and extract audio for each
    for i in xrange(n_samples):
        t = (ref_onsets[i] + ref_offsets[i])/2
        idx = np.floor((t-t_start) * sr)
        sample = y[idx-sample_len:idx+sample_len]
        librosa.output.write_wav(outdir+'/true_'+str(int(t*10))+'.wav', sample, sr)

    # Now produce samples that contain no calls
    for _ in xrange(n_samples):
        while True:     # Dangerous!
            idx = np.random.choice(len(y))
            t = idx*1.0/sr + t_start
            t_min = t-0.075
            t_max = t+0.075
            if t_min < t_start:
                continue
            if t_max > t_start + duration:
                continue
            # Check if this timepoint collides with annotation
            poop = ref_onsets[ref_onsets > t_min]
            poop = poop[poop < t_max]
            if np.size(poop) == 0:
                sample = y[idx-sample_len:idx+sample_len]
                librosa.output.write_wav(outdir+'/false_'+str(int(t*10))+'.wav', sample, sr)
                break
            # Else continue while loop and choose new idx

# Main
filename = '../../audio/ALFRED_20110924_183200.wav'
outdir = '../../audio/samples/ALFRED'
annotation_path = '../../annotations/ALFRED_20110924_183200.HAND_high_442NFCs_IDaek_EDIT_TO_INCLUDE_ALL.txt'
t_start = 0.0
duration = 1000
extract_samples(filename, outdir, annotation_path, t_start, duration)
