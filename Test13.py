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
modelI = lm('Murdock88.h5')
fsPrueba = 22000
n_fftPrueba = int(fsPrueba*0.025)
hop_lenghtPrueba = n_fftPrueba//2
fs = 22000
n_fft = int(fs*0.025)
hop_lenght = n_fft//2
palabra = ''
message_queue = queue.Queue()

recognizer = sr.Recognizer()
mic = sr.Microphone()
fs = 22000
n_fft = int(fs*0.025) #ES EL PRODUCTO DEL TIEMPO QUE SEPARA LOS FONOS PARA ASI HACER LOS BINS
hop_lenght = n_fft//2
modelI = lm('Beethoven80SISEÑOR.h5')

palabra =''


def predictionVector (audio):
   #Preprocesamiento del audio
     data2, sr = librosa.load(audio, mono=True, sr=fs)
     clip2 = librosa.effects.trim(data2, top_db= 20)
     audioNR2 = librosa.util.normalize(clip2[0])
     audioSR2 = nr.reduce_noise(audioNR2, fs)
     melS2 = librosa.feature.melspectrogram(y= audioSR2, sr=fs, n_mels=40, n_fft=n_fft, hop_length=hop_lenght)
     melP2= np.mean(melS2.T, axis=0)
     log_S2=librosa.power_to_db(melS2, ref=np.max)
     mfcc2 = librosa.feature.mfcc(S=melS2, n_mfcc=13)
     mfccP2= np.mean(mfcc2.T, axis=0)
     delta_mfcc2= librosa.feature.delta(mfccP2)
     delta2_mfcc2= librosa.feature.delta(mfccP2, order=2)
     chroma2 = librosa.feature.chroma_stft(S=log_S2, sr=fs)
     chromaP2 = np.mean(chroma2.T, axis=0)
     tonnetz2 = librosa.feature.tonnetz(y=librosa.effects.harmonic(data2),sr=fs)
     tonnetP2 = np.mean(tonnetz2.T,axis=0)

     arr22 = np.hstack([mfccP2, delta_mfcc2, delta2_mfcc2, chromaP2,melP2, tonnetP2])
     arr22 = np.expand_dims(arr22, axis=0)
     vector_predicted = modelI.predict(arr22)

     return vector_predicted

def WordRecognizer():
   while True:
    #SP_recognizer()
    print("Porfavor habla")
    with mic as source:
      audio = recognizer.listen(source, phrase_time_limit=3)
      with open('speech.wav', 'wb') as f:
        f.write(audio.get_wav_data())
    prediccion = predictionVector('speech.wav')
    etiquetas = ['Amarillo', 'Azul', 'Blanco', 'Rojo', 'Verde']
    palabra = etiquetas[prediccion.argmax(axis=1)[0]]
    #keyPress(palabra)

    print(palabra)
    remove("speech.wav")
    time.sleep(2)

""" def predictionVector(audio):
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
            with open('speech.wav', 'wb') as f:
                f.write(audio.get_wav_data())
        prediccion = predictionVector(audio)
        etiquetas = ['Amarillo', 'Azul', 'Blanco', 'Rojo', 'Verde']
        palabra = etiquetas[prediccion.argmax(axis=1)[0]]
        print(palabra)
        message_queue.put(palabra)
        time.sleep(2)

        print(palabra)
        remove('speech.wav') """