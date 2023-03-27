import arcade


import pickle
#import wave

import joblib as jb
from sklearn import datasets

import Test2 as beto
from multiprocessing import Process
import time

import speech_recognition as sr
import matplotlib.pyplot as plt #es la libreria para graficar
import numpy as np #libreria para operaciones matematicas
import noisereduce as nr #reduccion de ruido
#LIBROSA
import librosa
import librosa.display
from keras.models import load_model as lm
import time
import keyboard
from os import remove
#Constantes
Screen_width = 1000
Screen_height = 500
Screen_title = "Mario Signals"

#Constantes para escalar los sprites 
Character_scaling = 0.20
Ground_scaling = 0.20
Cylinder_scaling = 0.20

#Velocidad del jugador
PlayerMovement_speed = 5
Gravity = 1
Player_jump_speed = 15

#Pixeles mínimos y máximos de la pantalla
Left_viewport_margin = 250
Right_viewport_margin = 250
Bottom_viewport_margin = 50
Top_viewport_margin = 100

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
    keyPress(palabra)

    print(palabra)
    remove("speech.wav")
    time.sleep(2)

def press_key(key):
    keyboard.press(key)
    time.sleep(0.35) 
    keyboard.release(key)

def keyPress(palabra):
  if palabra =='Verde':
     press_key('UP')
  elif palabra =='Rojo':
     press_key('LEFT')
  elif palabra =='Azul':
    press_key('RIGHT')
    if palabra =='Blanco':
     press_key('UP')
     press_key('E')
  elif palabra =='Amarillo':
     press_key('UP')
     press_key('Q')
     

class Mygame(arcade.Window):
  def __init__(self):
    super().__init__(Screen_width,Screen_height,Screen_title)
    cosa = Process(target=WordRecognizer)
    cosa.start()
    arcade.set_background_color(arcade.csscolor.CORNFLOWER_BLUE)
    
    #Listas que contienen los sprites.
    self.coin_list = None
    self.wall_list = None
    self.player_list = None

    #Variable del  sprite del jugador.
    self.player_sprite = None

    #Se usa para mantener el scrolling
    self.view_bottom = 0
    self.view_left = 0

  
  def setup(self):
    self.coin_list = arcade.SpriteList()
    self.player_list = arcade.SpriteList()
    self.wall_list = arcade.SpriteList()
    

    #Crea el jugador.
    image_source = "amogus.png"
    self.player_sprite = arcade.Sprite(image_source, Character_scaling)
    self.player_sprite.center_x = 120
    self.player_sprite.center_y = 120
    self.player_list.append(self.player_sprite)

    #Se crea el piso.
    for x in range(0, 1250 , 64):
       wall = arcade.Sprite("wall.png", Ground_scaling)
       wall.center_x = x
       wall.center_y = 20
       self.wall_list.append(wall)

    #Se crean los obstaculos.
    coordinate_list = [[512, 135],
                       [215, 135],
                       [768, 135]]

    for coordinate in coordinate_list:
      #Inserta los obstaculos en el suelo
      wall = arcade.Sprite("Obstaculo1.png", Cylinder_scaling) 
      wall.position = coordinate
      self.wall_list.append(wall)

      #Declaramos el motor de física.
      self.physics_engine = arcade.PhysicsEnginePlatformer(self.player_sprite, self.wall_list, Gravity)

  
  def on_draw(self):
    arcade.start_render() 
    self.wall_list.draw()
    self.player_list.draw()

  def on_key_press(self, key, modifiers):
     
    if key == arcade.key.UP:
       if self.physics_engine.can_jump():
        self.player_sprite.change_y = Player_jump_speed
    elif key == arcade.key.RIGHT:
       self.player_sprite.change_x = -PlayerMovement_speed
    elif key == arcade.key.LEFT:
       self.player_sprite.change_x = PlayerMovement_speed
    

  def on_key_release(self, key, modifiers):
     if key == arcade.key.LEFT:
       self.player_sprite.change_x = 0
     elif key == arcade.key.RIGHT:
       self.player_sprite.change_x = 0

  def on_update(self, delta_time):
     self.physics_engine.update()
  
  
  

def main():
  window = Mygame()
  window.setup()
  arcade.run()


if __name__ == "__main__":
	main()