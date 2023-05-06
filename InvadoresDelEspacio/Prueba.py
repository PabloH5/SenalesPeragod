import time
from threading import Thread
import queue
import socket
import speech_recognition as sr
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import noisereduce as nr
import librosa
import librosa.display
from keras.models import load_model as lm
from librosa.core.spectrum import stft
from PIL import Image
from os import remove


recognizer = sr.Recognizer()
mic = sr.Microphone()
modelI = lm('MurdockLab93.h5')
fsPrueba = 22000
n_fftPrueba = int(fsPrueba*0.025)
hop_lenghtPrueba = n_fftPrueba//2
fs = 22000
n_fft = int(fs*0.025)
hop_lenght = n_fft//2
message_queue = queue.Queue()

def WordRecognizer():
    while True:
        palabra = ''
        print("Porfavor habla")
        with mic as source:
            audio = recognizer.listen(source, phrase_time_limit=1)
            with open('speech.wav', 'wb') as f:
                f.write(audio.get_wav_data())
            mic.stream.close
            
        data, sr = librosa.load('speech.wav', mono=True)
        clip2 = librosa.effects.trim(data, top_db=20)
        audioNR2 = librosa.util.normalize(clip2[0])
        audioSR2 = nr.reduce_noise(audioNR2, fsPrueba)
        plt.specgram(audioSR2, NFFT=n_fft, Fs=fs, Fc=0, cmap=plt.cm.jet, scale='dB')
        plt.axis('off')
        plt.savefig('audio.jpg')
        fileName = 'audio.jpg'
        imagen = Image.open(fileName)

        box = (80, 58, 576, 427)
        img2 = imagen.crop(box)
        imagenes = np.array(img2)
        arrayFinal = np.expand_dims(imagenes, axis=0)

        vector_predicted = modelI.predict(arrayFinal)
        etiquetas = ['Amarillo', 'Azul', 'Blanco', 'Rojo', 'Verde']
        palabra = etiquetas[vector_predicted.argmax(axis=1)[0]]
        print(palabra)
        # message_queue.put(palabra)
        
        time.sleep(1)

        remove('speech.wav')
        remove('audio.jpg')
    return palabra

WordRecognizer()
