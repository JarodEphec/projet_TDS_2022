import wave
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import fft
from scipy import signal
from scipy.fft import fftshift
from scipy.io import wavfile
import pyaudio
from datetime import datetime
import os

def recording():
    """This method allows the user to input their voice.
    """
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second
    seconds = 3
    filename = "output.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    os.system("clear")
    print('Parlez.')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

def file_to_process(file_name):
    """This method processes the wav file into a numpy array.
    """
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

def lissage(signal_brut, L):
    res = np.copy(signal_brut)  # duplication des valeurs

    for i in range(1, len(signal_brut) - 1):  # toutes les valeurs sauf la première et la dernière
        L_g = min(i, L)  # nombre de valeurs disponibles à gauche
        L_d = min(len(signal_brut) - i - 1, L)  # nombre de valeurs disponibles à droite
        Li = min(L_g, L_d)
        res[i] = np.sum(signal_brut[i - Li:i + Li + 1]) / (2 * Li + 1)

    return res

def cut_signal(signal_lisse):
    new_signal = []
    for value in range(len(signal_lisse)):
        if signal_lisse[value] >= 200:  # Valeur de l'amplitute a partir du quel on suppose que la personne parle
            new_signal.append(signal_lisse[value])
    return new_signal

def diff_signal():
    """This method checks the difference between the signals.
    """
    signal_absolue1 = np.absolute(file_to_process("output.wav")) 
    signal_lisse1 = lissage(signal_absolue1, 5000)

    signal_absolue2 = np.absolute(file_to_process("gab_final_output.wav")) 
    signal_lisse2 = lissage(signal_absolue2, 5000)

    new_signal1 = cut_signal(signal_lisse1)
    new_signal2 = cut_signal(signal_lisse2)

    if len(new_signal1) < len(new_signal2):
        new_signal2 = new_signal2[:len(new_signal1)]

    elif len(new_signal1) > len(new_signal2):
        new_signal1 = new_signal1[:len(new_signal2)]

    # transformation des deux signaux par CWT pour trouver leurs positions des piques
    peaks_1 = signal.find_peaks_cwt(new_signal1, np.arange(250, 500),
        max_distances = np.arange(250, 500) * 2)
    
    peaks_2 = signal.find_peaks_cwt(new_signal2, np.arange(250, 500),
        max_distances = np.arange(250, 500 ) * 2)

    # mise en commun des longueurs des arrays des piques
    m = min(len(peaks_1), len(peaks_2))
    peaks_1 = np.array(peaks_1[:m])
    peaks_2 = np.array(peaks_2[:m])

    print(peaks_1)
    print(peaks_2)

    # reconnaissance de personne par la position de pique
    peak_difference = []

    similar_peaks = 0
    for i in range(len(peaks_1)):
        peak_difference.append(abs(new_signal1[peaks_1[i]] - new_signal2[peaks_2[i]]))
        if (abs(peaks_1[i] - peaks_2[i]) <= 4000):
            similar_peaks = similar_peaks + 1

    if similar_peaks < 4:
        print("Not the same person")
    else:
        print("Hello, familiar person!")

    print(peak_difference)

    # mise en commun de l'amplitude des deux signaux par la moyenne de difference
    mean_peak_diff = np.mean(peak_difference)
    new_signal3 = [x + mean_peak_diff for x in new_signal1]

    signals_difference = []

    for value in range(len(new_signal3)):
        signals_difference.append(abs(new_signal3[value] - new_signal2[value]))
    
    plt.subplot(3, 1, 1)
    plt.title("Bonjour")
    plt.plot(new_signal1, color="blue")
    plt.scatter
    plt.ylabel("Amplitude")
    plt.xlabel("Temps")

    plt.subplot(3, 1, 2)
    plt.title("BonjourModifié")
    plt.plot(new_signal3, color="blue")
    plt.scatter
    plt.ylabel("Amplitude")
    plt.xlabel("Temps")

    plt.subplot(3, 1, 3)
    plt.title("BonjourDB")
    plt.plot(new_signal2, color="blue")
    plt.ylabel("Amplitude")
    plt.xlabel("Temps")
    plt.draw()

    return sum(signals_difference)/len(signals_difference)

def unlock():
    os.system("ssh root@tds \"loginctl unlock-sessions\"")
    print("System unlocked.")

if __name__ == '__main__':
    recording()

    result = diff_signal()
    if result <= 400:
        print(f"{result} authentification réussie.")
        unlock()
    else:
        print(f"{result} authentification ratée.")
    plt.show()
