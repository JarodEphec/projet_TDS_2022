import wave
import numpy as np
import sys
import matplotlib.pyplot as plt


def show_plot(file_name):
    if not file_name.endswith('.wav'):
        print("Audio avec l'extension wav uniquement.")
        sys.exit(0)
    wav = wave.open(file_name, "r")
    raw = wav.readframes(-1)
    raw = np.frombuffer(raw, "int16")

    if wav.getnchannels() == 2:
        print("Audio stéréo non supporter.")
        sys.exit(0)
    plt.title("Bonjour")
    plt.plot(raw, color="blue")
    plt.ylabel("Amplitude")
    plt.xlabel("Temps")
    plt.show()


if __name__ == '__main__':
    show_plot('bonjour.wav')
