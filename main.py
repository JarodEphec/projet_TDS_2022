import wave
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import fft


def show_plot(file_name):
    raw = file_to_process(file_name)
    plt.title("Bonjour")
    plt.plot(raw, color="blue")
    plt.ylabel("Amplitude")
    plt.xlabel("Temps")


def show_plot_raw(raw):
    plt.title("Bonjour")
    plt.plot(raw, color="black")
    plt.ylabel("Amplitude")
    plt.xlabel("Temps")


def file_to_process(file_name):
    if not file_name.endswith('.wav'):
        print("Audio avec l'extension wav uniquement.")
        sys.exit(0)
    wav = wave.open(file_name, "r")
    raw = wav.readframes(-1)
    raw = np.frombuffer(raw, "int16")
    if wav.getnchannels() == 2:
        print("Audio stéréo non supporter.")
        sys.exit(0)
    return raw


def compare_two_signal(signal1, signal2):

    audio1_as_np_int16 = file_to_process(signal1)
    audio2_as_np_int16 = file_to_process(signal2)
    result = 0
    if len(audio2_as_np_int16) != len(audio1_as_np_int16):
        print("Vos deux son n'ont pas la même longueur.")
        sys.exit(0)
    for pos in range(len(audio2_as_np_int16)):
        if max(audio1_as_np_int16[pos] - audio2_as_np_int16[pos], audio2_as_np_int16[pos] - audio1_as_np_int16[pos]) == 0:
            result += 1
    return result / len(audio2_as_np_int16) * 100

def compute_similarity(ref_rec,input_rec):
    ## Time domain similarity
    ref_time = np.correlate(ref_rec, ref_rec)
    inp_time = np.correlate(ref_rec, input_rec)
    diff_time = abs(ref_time-inp_time)

    ## Freq domain similarity
    ref_freq = np.correlate(np.fft.fft(ref_rec), np.fft.fft(ref_rec))
    inp_freq = np.correlate(np.fft.fft(ref_rec), np.fft.fft(input_rec))
    diff_freq = abs(ref_freq-inp_freq)

    ## Power similarity
    ref_power = np.sum(ref_rec**2)
    inp_power = np.sum(input_rec**2)
    diff_power = abs(ref_power-inp_power)

    return float(diff_time), float(diff_freq), float(diff_power)


if __name__ == '__main__':
    print("similitude de ", compare_two_signal(
        'bonjour.wav', 'gab_bonjour_1.wav'), ' %')
    plt.subplot(3, 1, 1)
    show_plot('bonjour.wav')
    plt.subplot(3, 1, 2)
    show_plot('patate.wav')
    plt.subplot(3, 1, 3)
    raw = np.correlate(file_to_process(
        'bonjour.wav'), file_to_process('patate.wav'), 'same')
    print("Cross correlation : ", np.average(raw))
    show_plot_raw(raw)
    plt.show()
    corr = fft.ifft(fft.fft(file_to_process('bonjour.wav')) * np.conj(fft.fft(file_to_process('patate.wav'))))
    print(np.average(corr))
    print(compute_similarity(file_to_process('bonjour.wav'), file_to_process('bonjour.wav')))
    print(compute_similarity(file_to_process('bonjour.wav'), file_to_process('patate.wav')))
    print(compute_similarity(file_to_process('bonjour.wav'), file_to_process('bonjour2.wav')))
