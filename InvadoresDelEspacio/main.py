#Librerias
import pygame
import random
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
from multiprocessing import Process


recognizer = sr.Recognizer()
mic = sr.Microphone()
modelI = lm('InvadoresDelEspacio\MurdockLab93.h5')
fsPrueba = 22000
n_fftPrueba = int(fsPrueba*0.025)
hop_lenghtPrueba = n_fftPrueba//2
fs = 22000
n_fft = int(fs*0.025)
hop_lenght = n_fft//2
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


   
    
pygame.init()
pygame.mixer.init()
#Se cargan los recursos multimediales
fondo = pygame.image.load('InvadoresDelEspacio/imagenes/bgimg.png')
laser_sonido = pygame.mixer.Sound('InvadoresDelEspacio/laser.wav')
explosion_sonido = pygame.mixer.Sound('InvadoresDelEspacio/explosion.wav')
golpe_sonido = pygame.mixer.Sound('InvadoresDelEspacio/golpe.wav')

explosion_list = []
for i in range(1,13):
	explosion = pygame.image.load(f'InvadoresDelEspacio/explosion/{i}.png')
	explosion_list.append(explosion)
	
width = fondo.get_width()
height = fondo.get_height()
window = pygame.display.set_mode((width, height))
pygame.display.set_caption('Juego InvasoresDelEspacio')
run = True
fps = 60
clock = pygame.time.Clock()
score = 0
vida = 100
blanco = (255,255,255)
negro = (0,0,0)
#Se definen los labels de puntuación y barra de vida
def texto_puntuacion(frame, text, size, x,y):
	font = pygame.font.SysFont('Small Fonts', size, bold=True)
	text_frame = font.render(text, True, blanco,negro)
	text_rect = text_frame.get_rect()
	text_rect.midtop = (x,y)
	frame.blit(text_frame, text_rect)

def barra_vida(frame, x,y, nivel):
	longitud = 100
	alto = 20
	fill = int((nivel/100)*longitud)
	border = pygame.Rect(x,y, longitud, alto)
	fill = pygame.Rect(x,y,fill, alto)
	pygame.draw.rect(frame, (255,0,55),fill)
	pygame.draw.rect(frame, negro, border,4)


class Jugador(pygame.sprite.Sprite):
	def __init__(self):
		super().__init__()
		self.image = pygame.image.load('InvadoresDelEspacio/imagenes/A1.png').convert_alpha()
		pygame.display.set_icon(self.image)
		self.rect = self.image.get_rect()
		self.rect.centerx = width//2
		self.rect.centery = height-50
		self.velocidad_x = 0
		self.vida = 100

	def update(self):
		self.velocidad_x = 0
		self.velocidad_y = 0
		keystate = pygame.key.get_pressed()
		if keystate[pygame.K_LEFT]:
			self.velocidad_x = -5
		elif keystate[pygame.K_RIGHT]:
			self.velocidad_x = 5
		elif keystate[pygame.K_UP]:
			self.velocidad_y = -5
		elif keystate[pygame.K_DOWN]:
			self.velocidad_y = 5

		self.rect.x += self.velocidad_x
		if self.rect.right > width:
			self.rect.right = width
		elif self.rect.left < 0:
			self.rect.left = 0

		self.rect.y += self.velocidad_y
		if self.rect.bottom > height:
			self.rect.bottom = height
		elif self.rect.top < 100:
			self.rect.top = 100

	def disparar(self):
		bala = Balas(self.rect.centerx, self.rect.top)
		grupo_jugador.add(bala)
		grupo_balas_jugador.add(bala)
		laser_sonido.play()

class Enemigos(pygame.sprite.Sprite):
	def __init__(self, x, y):
		super().__init__()
		self.image = pygame.image.load('InvadoresDelEspacio/imagenes/E1.png').convert_alpha()
		self.rect = self.image.get_rect()
		self.rect.x = random.randrange(1, width-50)
		self.rect.y = 10
		self.velocidad_y = random.randrange(-5,20)

	def update(self):
		self.time = random.randrange(-1, pygame.time.get_ticks()//5000)
		self.rect.x += self.time
		if self.rect.x >= width:
			self.rect.x = 0
			self.rect.y += 50

	def disparar_enemigos(self):
		bala = Balas_enemigos(self.rect.centerx, self.rect.bottom)
		grupo_jugador.add(bala)
		grupo_balas_enemigos.add(bala)
		laser_sonido.play()

class Balas(pygame.sprite.Sprite):
	def __init__(self, x, y):
		super().__init__()
		self.image = pygame.image.load('InvadoresDelEspacio/imagenes/B2.png').convert_alpha()
		self.rect = self.image.get_rect()
		self.rect.centerx = x
		self.rect.y = y
		self.velocidad = -18

	def update(self):
		self.rect.y +=  self.velocidad
		if self.rect.bottom <0:
			self.kill()

class Balas_enemigos(pygame.sprite.Sprite):
	def __init__(self, x, y):
		super().__init__()
		self.image = pygame.image.load('InvadoresDelEspacio/imagenes/B1.png').convert_alpha()
		self.image = pygame.transform.rotate(self.image, 180)
		self.rect = self.image.get_rect()
		self.rect.centerx = x 
		self.rect.y = random.randrange(10, 40)
		self.velocidad_y = 1

	def update(self):
		self.rect.y +=  self.velocidad_y 
		if self.rect.bottom > height:
			self.kill()

class Explosion(pygame.sprite.Sprite):
	def __init__(self, position):
		super().__init__()
		self.image = explosion_list[0]	
		img_scala = pygame.transform.scale(self.image, (20,20))	
		self.rect = img_scala.get_rect()
		self.rect.center = position
		self.time = pygame.time.get_ticks()
		self.velocidad_explo = 30
		self.frames = 0 
		
	def update(self):
		tiempo = pygame.time.get_ticks()
		if tiempo - self.time > self.velocidad_explo:
			self.time = tiempo 
			self.frames+=1
			if self.frames == len(explosion_list):
				self.kill()
			else:
				position = self.rect.center
				self.image = explosion_list[self.frames]
				self.rect = self.image.get_rect()
				self.rect.center = position

grupo_jugador = pygame.sprite.Group()
grupo_enemigos = pygame.sprite.Group()
grupo_balas_jugador = pygame.sprite.Group()
grupo_balas_enemigos = pygame.sprite.Group()

player = Jugador()
grupo_jugador.add(player)
grupo_balas_jugador.add(player)

for x in range(10):
	enemigo = Enemigos(10,10)
	grupo_enemigos.add(enemigo)
	grupo_jugador.add(enemigo)

while run:
	clock.tick(fps)
	window.blit(fondo, (0,0))

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False
		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_SPACE:
				player.disparar()

	grupo_jugador.update()
	grupo_enemigos.update()
	grupo_balas_jugador.update()
	grupo_balas_enemigos.update() 

	grupo_jugador.draw(window)

    # Logica coliciones balas del jugador hacia el enemigo

	colicion1 = pygame.sprite.groupcollide(grupo_enemigos, grupo_balas_jugador,True,True)
	for i in colicion1:	
		score+=10
		enemigo.disparar_enemigos()
		enemigo = Enemigos(300,10)
		grupo_enemigos.add(enemigo)
		grupo_jugador.add(enemigo)

		explo = Explosion(i.rect.center)
		grupo_jugador.add(explo)
		explosion_sonido.set_volume(0.3)		
		explosion_sonido.play()

	# Logica coliciones balas del enemigo hacia el jugador 

	colicion2 = pygame.sprite.spritecollide(player, grupo_balas_enemigos, True)
	for j in colicion2:
		player.vida -= 10
		if player.vida <=0:
			run = False
		explo1 = Explosion(j.rect.center)
		grupo_jugador.add(explo1)
		golpe_sonido.play()  

	if score >= 100:
		mensaje_ganador = pygame.font.Font(None, 100).render("¡Ganaste!", True, blanco)
		mensaje_rect = mensaje_ganador.get_rect()
		mensaje_rect.centerx = window.get_rect().centerx
		mensaje_rect.centery = window.get_rect().centery
		window.blit(mensaje_ganador, mensaje_rect)
		pygame.display.update()
		pygame.time.wait(2000)

	# Se muestra el indicador y score
	texto_puntuacion(window, ('  SCORE: '+ str(score)+'       '), 30, width-85, 2)
	barra_vida(window, width-285, 0, player.vida)
	

	pygame.display.flip()
pygame.quit()
