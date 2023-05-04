import pyaudio
import numpy as np
import tensorflow as tf
import speech_recognition as sr
import matplotlib.pyplot as plt #es la libreria para graficar
import numpy as np #libreria para operaciones matematicas
import noisereduce as nr #reduccion de ruido
#LIBROSA
import librosa
import librosa.display
from keras.models import load_model as lm
import time
from os import remove
from librosa.core.spectrum import stft
from PIL import Image

modelI = lm('Murdock88.h5')

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()

fs = 22000
n_fft = int(fs*0.025) #ES EL PRODUCTO DEL TIEMPO QUE SEPARA LOS FONOS PARA ASI HACER LOS BINS
hop_lenght = n_fft//2

def predictionVector (audio):
   #Preprocesamiento del audio
    data2, sr = librosa.load(audio, mono=True, sr=fs)
    clip2 = librosa.effects.trim(data2, top_db= 20)
    audioNR2 = librosa.util.normalize(clip2[0])
    audioSR2 = nr.reduce_noise(audioNR2, fs)
    plt.specgram(audioSR2,NFFT=n_fft,Fs=fs,Fc=0,cmap=plt.cm.jet,scale='dB')
    plt.axis('off')
    plt.savefig('audio.jpg')
    fileName='audio.jpg'
    imagen = Image.open(fileName)
    box = (80,58,576,427)
    img2 = imagen.crop(box)
    # Agregar la imagen a la lista de imágenes
    imagenes=np.array(img2)
    arrayFinal = np.expand_dims(imagenes, axis=0)
    vector_predicted = modelI.predict(arrayFinal)

    return vector_predicted

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=predictionVector)

stream.start_stream()

while stream.is_active():
    try:
        time.sleep(0.1)
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()

