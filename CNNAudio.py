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


recognizer = sr.Recognizer()
mic = sr.Microphone()
modelI = lm('Murdock88.h5')
fsPrueba = 22000
n_fftPrueba = int(fsPrueba*0.025)
hop_lenghtPrueba = n_fftPrueba//2
fs = 22000
n_fft = int(fs*0.025)
hop_lenght = n_fft//2
palabra = ''
message_queue = queue.Queue()

def predictionVector(audio):
    data, sr = librosa.load(audio, mono=True)
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

    return vector_predicted

def WordRecognizer():
    while True:
        print("Porfavor habla")
        with mic as source:
            audio = recognizer.listen(source, phrase_time_limit=3)
        prediccion = predictionVector(audio)
        etiquetas = ['Amarillo', 'Azul', 'Blanco', 'Rojo', 'Verde']
        palabra = etiquetas[prediccion.argmax(axis=1)[0]]
        print(palabra)
        message_queue.put(palabra)
        time.sleep(2)


def SocketConnection():
    HOST = '127.0.0.1'
    PORT = 65432

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        while True:
            if not message_queue.empty():
                data = message_queue.get()
                s.sendall(data.encode())
                time.sleep(1)

# Crea los hilos
thread1 = Thread(target=WordRecognizer)
thread2 = Thread(target=SocketConnection)

# Inicia los hilos
thread1.start()
thread2.start()

# Espera a que los hilos terminen
thread1.join()
thread2.join()
