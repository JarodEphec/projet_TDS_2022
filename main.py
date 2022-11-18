import wave
import numpy as np
import sys
import matplotlib.pyplot as plt


def show_plot(file_name):
    raw = file_to_process(file_name)
    plt.title("Bonjour")
    plt.plot(raw, color="blue")
    plt.ylabel("Amplitude")
    plt.xlabel("Temps")
    plt.show()


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


if __name__ == '__main__':
    show_plot('bonjour.wav')
    print("similitude de ", compare_two_signal(
        'bonjour.wav', 'bonjour2.wav'), ' %')
