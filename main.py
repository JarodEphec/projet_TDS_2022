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
from copy import copy

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

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], analog=False, btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def cut_signal(signal_lisse):
    new_signal = []
    for value in range(len(signal_lisse)):
        if signal_lisse[value] >= 200:  # Valeur de l'amplitute a partir du quel on suppose que la personne parle
            new_signal.append(signal_lisse[value])
    return new_signal

def filter_signal(signal):
    sample_rate = 44100.0
    human_low_freq = 70.0
    human_high_freq = 270.0

    filtered_signal = butter_bandpass_filter(signal, human_low_freq, human_high_freq, sample_rate)

def diff_signal():
    """This method checks the difference between the signals.
    """
    #signal_brut1 = file_to_process("output.wav")
    signal_absolue1 = np.absolute(file_to_process("output.wav")) 
    #signal_filtre1 = filter_signal(signal_absolue1)
    signal_lisse1 = lissage(signal_absolue1, 5000)

    #signal_brut2 = file_to_process("gab_final_output.wav")
    signal_absolue2 = np.absolute(file_to_process("gab_final_output.wav")) 
    #signal_filtre2 = filter_signal(signal_absolue2)
    signal_lisse2 = lissage(signal_absolue2, 5000)

    #signal1 = signal.savgol_filter(signal_lisse1, 53, 3, mode='nearest')
    #signal2 = signal.savgol_filter(signal_lisse2, 53, 3, mode='nearest')

    new_signal1 = cut_signal(signal_lisse1)
    new_signal2 = cut_signal(signal_lisse2)

    #new_signal1 = cut_signal(signal1)
    #new_signal2 = cut_signal(signal2)

    # mise en commun du longueur des signaux
    if len(new_signal1) < len(new_signal2):
        new_signal2 = new_signal2[:len(new_signal1)]

    elif len(new_signal1) > len(new_signal2):
        new_signal1 = new_signal1[:len(new_signal2)]

    #sgsav1 = signal.savgol_filter(new_signal1, 51, 3)
    #sgsav2 = signal.savgol_filter(new_signal2, 51, 3)

    peaks_1 = signal.find_peaks_cwt(new_signal1, np.arange(250, 500),
        max_distances = np.arange(250, 500) * 2)
    
    peaks_2 = signal.find_peaks_cwt(new_signal2, np.arange(250, 500),
        max_distances = np.arange(250, 500 ) * 2)

    # match la longueur des deux indexes de peak
    m = min(len(peaks_1), len(peaks_2))
    peaks_1 = np.array(peaks_1[:m])
    peaks_2 = np.array(peaks_2[:m])

    print(peaks_1)
    print(peaks_2)

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

    mean_peak_diff = np.mean(peak_difference)
    new_signal3 = [x + mean_peak_diff for x in new_signal1]

    signals_difference = []

    for value in range(len(new_signal1)):
        signals_difference.append(abs(new_signal1[value] - new_signal2[value]))
    
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
    #os.system("ssh root@tds \"loginctl unlock-sessions\"")
    print("System unlocked.")

if __name__ == '__main__':
    #recording()

    result = diff_signal()
    if result <= 400:
        print(f"{result} authentification réussie.")
        unlock()
    else:
        print(f"{result} authentification ratée.")
    plt.show()
