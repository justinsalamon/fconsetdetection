import librosa


def trim_audio(filename, start_time, end_time):
    """
    Extracts a selection of audio
    :param filename: string
    :param start_time: float (seconds)
    :param end_time:  float (seconds)
    :return: new filename
    """

    # load selectrion of audio
    y, sr = librosa.core.load(filename, sr=None, offset=start_time, duration=end_time-start_time)
    # rename
    i = filename.rfind(".")
    out_filename = filename[:i] + "_{}-{}".format(start_time, end_time) + filename[i:]
    # save
    librosa.output.write_wav(out_filename, y, sr)

    return out_filename

trim_audio("../../audio/ALFRED_20110924_183200.wav", 0, 3600)