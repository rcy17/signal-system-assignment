# coding: utf-8
from scipy.io import wavfile
import numpy as np
from scipy.signal import windows
from scipy.fftpack import fft
import os


def get_standard_tune(file):
    # read wav first
    frame_per_second, data = wavfile.read(file)
    if len(data.shape) == 2:
        data = data[:, 0]

    # then set some basic factors
    frequency_resolution = 1
    slice_length = frame_per_second // frequency_resolution
    n_fft = slice_length
    interval = slice_length // 4
    frame_number = (len(data) - slice_length) // interval + 1
    fft_amplitude = np.zeros([n_fft // 2 + 1, frame_number])
    hamming = windows.hamming(slice_length)

    # process fft for each frame
    for frame in range(frame_number):
        frame_start = interval * frame
        data_slice = data[frame_start:frame_start + slice_length]
        data_windowed = data_slice * hamming
        fft_result = fft(data_windowed, slice_length) / slice_length
        # fft result's amplitude is symmetric, so we just get half data of result
        # by the way, plus 1e-16 to avoid -inf warning in log function
        fft_amplitude[:, frame] = 2 * abs(fft_result[:n_fft // 2 + 1]) + 1e-16
    # use log to suppress the amplitude
    fft_amplitude_log = np.log(fft_amplitude)

    # expect a4 tune to be 339 to 445, which is prior
    a4 = np.arange(439, 446)
    # a4 to g3: 2^(-14/12), a4 to c7: 2^(27/12)
    coefficient = np.power(2, np.arange(-14, 28) / 12)
    # here we use around to get closest frequency point
    mat = np.around(np.outer(a4, coefficient) / frequency_resolution).astype(int)
    # notice that score[0] for 339Hz, score[-1] is score[6] for 445Hz
    score = np.zeros(len(a4))

    # set lower limit for amplitude, here just omit negative number
    # otherwise -inf can destroy the analysis
    min_amp = 0
    # now sum up for each frequency
    for i in range(len(score)):
        aim = fft_amplitude_log[mat[i]]
        # here divide frame_number to normalize the score
        score[i] = aim[aim > min_amp].sum() / frame_number
    # avoid overflow in softmax
    score -= score.max()
    # use softmax to evaluate probability
    probability = np.exp(score) / np.exp(score).sum()

    # then output the result
    for i in range(len(probability)):
        print("{}Hz: {:.2f}%".format(a4[i], probability[i] * 100))
    return


def check_file(file):
    return './data/' + file if os.path.isfile('./data/' + file) else file


def main():
    while True:
        try:
            file_name = input('input file name(only wav format supported, input "exit" to exit)\n')
            if file_name == 'exit':
                break
            get_standard_tune(check_file(file_name))
        except Exception as e:
            print(e)
    return 0


if __name__ == '__main__':
    main()
