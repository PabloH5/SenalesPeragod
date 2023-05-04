

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
import pyaudio
import io
import tempfile
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write

fsPrueba = 22000
n_fftPrueba  = int(fsPrueba*0.025) #ES EL PRODUCTO DEL TIEMPO QUE SEPARA LOS FONOS PARA ASI HACER LOS BINS
hop_lenghtPrueba = n_fftPrueba//2
fs = 22000
n_fft  = int(fs*0.025) #ES EL PRODUCTO DEL TIEMPO QUE SEPARA LOS FONOS PARA ASI HACER LOS BINS
hop_lenght = n_fft//2
duration = 2  # Duración de la grabación en segundos
modelI = lm('Murdock88.h5')

# Configuración de la grabación de audio
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22000
CHUNK = 1024

# Inicializar PyAudio
audio = pyaudio.PyAudio()

# Abrir el flujo de audio
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

print("Grabando...")



def convertData():
    data = stream.read(CHUNK, exception_on_overflow=False)
    wavaData = data.get_wav_data()
    numpyArray = np.frombuffer(wavaData, dtype=np.int16)
    numpyArray = numpyArray.astype(np.float64)
    audioPredict = predictionVector(numpyArray)
    etiquetas = ['Amarillo', 'Azul', 'Blanco', 'Rojo', 'Verde']
    palabra = etiquetas[audioPredict.argmax(axis=1)[0]]
    print(palabra)
    # keyPress(palabra)
    return audioPredict


def predictionVector (audio):
   
    data, sr = librosa.load(audio, mono=True)
    clip2 = librosa.effects.trim(data, top_db= 20)
    audioNR2 = librosa.util.normalize(clip2[0])
    audioSR2 = nr.reduce_noise(audioNR2, fsPrueba)
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
    # print(palabra)
    remove("speech.wav")
    remove("audio.jpg")
    return vector_predicted

    
    


n = True
while n==True:
   convertData()
