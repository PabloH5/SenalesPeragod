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
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import scout_apm
from scout_apm.api import Config

# Configurar Scout APM
config = Config(
    key="YOUR_SCOUT_APM_KEY",
    monitor=True
)
scout_apm.install(config)



modelI = lm('Murdock88.h5')

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()

fsPrueba = 22000
n_fftPrueba  = int(fsPrueba*0.025) #ES EL PRODUCTO DEL TIEMPO QUE SEPARA LOS FONOS PARA ASI HACER LOS BINS
hop_lenghtPrueba = n_fftPrueba//2
fs = 22000
n_fft  = int(fs*0.025) #ES EL PRODUCTO DEL TIEMPO QUE SEPARA LOS FONOS PARA ASI HACER LOS BINS
hop_lenght = n_fft//2
duration = 2  # Duración de la grabación en segundos

def predictionVector ():
   #Preprocesamiento del audio
    print("Grabando audio...")
    myrecording = sd.rec(int(fs * duration), samplerate=fs, channels=1)
    sd.wait()  # Esperar a que termine la grabación
    time.sleep(2)
    sf.write('myrecording.wav', myrecording, fs)
    data, sr = librosa.load("myrecording.wav", mono=True)
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
    remove("myrecording.wav")
    # keyPress(vector_predicted)
    return vector_predicted


# Configurar Scout APM
config = Config(
    key="YOUR_SCOUT_APM_KEY",
    monitor=True
)
scout_apm.install(config)

def my_function():
    with scout_apm.instrument("my_transaction"):
        # Llamar la función predictionVector() dentro de la transacción
        result = predictionVector()
        return result
    
my_function()