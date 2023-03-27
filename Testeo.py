import matplotlib.pyplot as plt #es la libreria para graficar
import numpy as np #libreria para operaciones matematicas
import noisereduce as nr #reduccion de ruido
#LIBROSA
import librosa
import librosa.display
fs = 22000
n_fft  = int(fs*0.025)
fsPrueba = 22000
n_fftPrueba  = int(fs*0.025) #ES EL PRODUCTO DEL TIEMPO QUE SEPARA LOS FONOS PARA ASI HACER LOS BINS
hop_lenghtPrueba = n_fft//2

#print(hop_lenght)
from keras.models import load_model as lm

modelI = lm('Beethoven80SISEÑOR.h5')
def prediction (audio2):
    arr3=[]
  #Preprocesamiento del audio
    data2, sr = librosa.load(audio2, mono=True, sr=fsPrueba)
    clip2 = librosa.effects.trim(data2, top_db= 20)
    audioNR2 = librosa.util.normalize(clip2[0])
    audioSR2 = nr.reduce_noise(audioNR2, fs)
    melS2 = librosa.feature.melspectrogram(y= audioSR2, sr=fsPrueba, n_mels=40, n_fft=n_fftPrueba, hop_length=hop_lenghtPrueba)
    melP2= np.mean(melS2.T, axis=0)
    log_S2=librosa.power_to_db(melS2, ref=np.max)
    mfcc2 = librosa.feature.mfcc(S=melS2, n_mfcc=13)
    mfccP2= np.mean(mfcc2.T, axis=0)
    delta_mfcc2= librosa.feature.delta(mfccP2)
    delta2_mfcc2= librosa.feature.delta(mfccP2, order=2)
    chroma2 = librosa.feature.chroma_stft(S=log_S2, sr=fs)
    chromaP2 = np.mean(chroma2.T, axis=0)
    tonnetz2 = librosa.feature.tonnetz(y=librosa.effects.harmonic(data2),sr=fsPrueba)
    tonnetP2 = np.mean(tonnetz2.T,axis=0)

    arr22 = np.hstack([mfccP2, delta_mfcc2, delta2_mfcc2, chromaP2,melP2, tonnetP2])
    arr22 = np.expand_dims(arr22, axis=0)

    vector_predicted = modelI.predict(arr22)
    
    return vector_predicted


prediccion = prediction('Rojo.6.wav')
print(prediccion.argmax(axis=1))
print(prediccion)
etiquetas = ['Amarillo', 'Azul', 'Blanco', 'Rojo', 'Verde']
print(f'Se predijo la palabra : {etiquetas[prediccion.argmax(axis=1)[0]]}')