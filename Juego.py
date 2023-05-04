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
import threading
import queue


from CNNAudio import CNNAudio
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
  elif palabra =='Blanco':
     press_key('UP')
     press_key('LEFT')
  elif palabra =='Amarillo':
     press_key('UP')
     press_key('RIGHT')
     

class Mygame(arcade.Window):
  def __init__(self):
    super().__init__(Screen_width, Screen_height, Screen_title)
    self.message_queue = queue.Queue()
    self.is_running = True
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

def process_message(self, message):
    # Lógica para procesar el mensaje recibido
    keyPress(message)

def queue_listener(self):
    while self.is_running:
        if not self.message_queue.empty():
            message = self.message_queue.get()
            self.process_message(message)

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
    self.queue_thread = threading.Thread(target=self.queue_listener)
    self.queue_thread.start()
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
def on_close(self):
    # Detiene el hilo al cerrar la ventana
    self.is_running = False
    self.queue_thread.join()
    super().on_close()

def on_update(self, delta_time):
    # Revisa si hay mensajes en la cola
    while not self.message_queue.empty():
        message = self.message_queue.get()
        self.process_message(message)
  
  
  

def main():
  window = Mygame()
  window.setup()
  arcade.run()


if __name__ == "__main__":
	main()