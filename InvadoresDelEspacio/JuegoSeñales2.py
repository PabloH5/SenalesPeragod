import random

import arcade
import speech_recognition as sr
import matplotlib.pyplot as plt
import numpy as np
import noisereduce as nr
import librosa
import librosa.display
import noisereduce as nr
import librosa
import librosa.display
from keras.models import load_model as lm
from librosa.core.spectrum import stft
from PIL import Image
from os import remove
from multiprocessing import Process
import time
import keyboard


SPRITE_SCALING_PLAYER = 0.8
SPRITE_SCALING_enemy = 0.8
SPRITE_SCALING_LASER = 0.8

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Slime Invaders"

BULLET_SPEED = 1.5
BULLER_PLAYER_SPEED = 3
ENEMY_SPEED = 0.5
PLAYER_SPEED = 35


# This margin controls how close the enemy gets to the left or right side
# before reversing direction.
ENEMY_VERTICAL_MARGIN = 10
RIGHT_ENEMY_BORDER = SCREEN_WIDTH - ENEMY_VERTICAL_MARGIN
LEFT_ENEMY_BORDER = ENEMY_VERTICAL_MARGIN

# How many pixels to move the enemy down when reversing
ENEMY_MOVE_DOWN_AMOUNT = 30

# Game state
GAME_OVER = 1
PLAY_GAME = 0
WIN = 2


recognizer = sr.Recognizer()
mic = sr.Microphone()
modelI = lm('InvadoresDelEspacio\MurdockLab94.h5')
fsPrueba = 22000
n_fftPrueba = int(fsPrueba*0.025)
hop_lenghtPrueba = n_fftPrueba//2
fs = 22000
n_fft = int(fs*0.025)
hop_lenght = n_fft//2

def predictionVector ():
    data, sr = librosa.load('speech.wav', mono=True)
    clip2 = librosa.effects.trim(data, top_db=20)
    audioNR2 = librosa.util.normalize(clip2[0])
    audioSR2 = nr.reduce_noise(audioNR2, fsPrueba)
    plt.specgram(audioSR2, NFFT=n_fft, Fs=fs, Fc=0, cmap=plt.cm.jet, scale='dB')
    plt.axis('off')
    plt.savefig('saudio.jpg')
    fileName = 'saudio.jpg'
    imagen = Image.open(fileName)

    box = (80, 58, 576, 427)
    img2 = imagen.crop(box)
    imagenes = np.array(img2)
    arrayFinal = np.expand_dims(imagenes, axis=0)
    vector_predicted = modelI.predict(arrayFinal)

    return vector_predicted

def WordRecognizer():
   while True:
    #SP_recognizer()
    print("Porfavor habla")
    with mic as source:
      audio = recognizer.listen(source, phrase_time_limit=1)
      with open('speech.wav', 'wb') as f:
        f.write(audio.get_wav_data())
    prediccion = predictionVector()
    etiquetas = ['Amarillo', 'Azul', 'Blanco', 'Rojo', 'Verde']
    palabra = etiquetas[prediccion.argmax(axis=1)[0]]
    keyPress(palabra)

    print(palabra)
    remove("speech.wav")
    remove("saudio.jpg")
    time.sleep(1)
# palabra = WordRecognizer()
def press_key(key):
    keyboard.press(key)
    time.sleep(0.5) 
    keyboard.release(key)

def keyPress(palabra):
  if palabra =='Verde':
     press_key('SPACE')
  elif palabra =='Rojo':
     press_key('LEFT')
  elif palabra =='Azul':
    press_key('RIGHT')
  elif palabra =='Blanco':
     press_key('UP')
  elif palabra =='Amarillo':
     press_key('DOWN')



class MyGame(arcade.Window):
    """ Main application class. """

    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        cosa = Process(target=WordRecognizer)
        cosa.start()
        # arcade.set_background_color(arcade.csscolor.CORNFLOWER_BLUE)
        # Variables that will hold sprite lists
        self.player_list = None
        self.enemy_list = None
        self.player_bullet_list = None
        self.enemy_bullet_list = None
        self.shield_list = None

        # Textures for the enemy
        self.enemy_textures = None

        # State of the game
        self.game_state = PLAY_GAME

        # Set up the player info
        self.player_sprite = None
        self.score = 0

        # Enemy movement
        self.enemy_change_x = -ENEMY_SPEED

        # Don't show the mouse cursor
        self.set_mouse_visible(True)

        # Load sounds. Sounds from kenney.nl
        self.gun_sound = arcade.load_sound(":resources:sounds/hurt5.wav")
        self.hit_sound = arcade.load_sound(":resources:sounds/hit5.wav")

        # arcade.set_background_color(arcade.color.AMAZON)
        
        self.background = arcade.load_texture("InvadoresDelEspacio\imagenes\Espacio.jpg")
        # arcade.configure_logging()

    def setup_level_one(self):
        # Load the textures for the enemies, one facing left, one right
        self.enemy_textures = []
        texture = arcade.load_texture("InvadoresDelEspacio\imagenes\E1.png", mirrored=True)
        self.enemy_textures.append(texture)
        texture = arcade.load_texture("InvadoresDelEspacio\imagenes\E1.png")
        self.enemy_textures.append(texture)

        # Create rows and columns of enemies
        x_count = 4
        x_start = 380
        x_spacing = 80
        y_count = 1
        y_start = 420
        y_spacing = 60
        for x in range(x_start, x_spacing * x_count + x_start, x_spacing):
            for y in range(y_start, y_spacing * y_count + y_start, y_spacing):

                # Create the enemy instance
                enemy = arcade.Sprite()
                enemy.scale = SPRITE_SCALING_enemy
                enemy.texture = self.enemy_textures[1]

                # Position the enemy
                enemy.center_x = x
                enemy.center_y = y

                # Add the enemy to the lists
                self.enemy_list.append(enemy)

    def make_shield(self, x_start):
        """
        Make a shield, which is just a 2D grid of solid color sprites
        stuck together with no margin so you can't tell them apart.
        """
        shield_block_width = 5
        shield_block_height = 10
        shield_width_count = 20
        shield_height_count = 5
        y_start = 150
        for x in range(x_start,
                       x_start + shield_width_count * shield_block_width,
                       shield_block_width):
            for y in range(y_start,
                           y_start + shield_height_count * shield_block_height,
                           shield_block_height):
                shield_sprite = arcade.SpriteSolidColor(shield_block_width,
                                                        shield_block_height,
                                                        arcade.color.WHITE)
                shield_sprite.center_x = x
                shield_sprite.center_y = y
                self.shield_list.append(shield_sprite)

    def setup(self):
        """
        Set up the game and initialize the variables.
        Call this method if you implement a 'play again' feature.
        """

        self.game_state = PLAY_GAME

        # Sprite lists
        self.player_list = arcade.SpriteList()
        self.enemy_list = arcade.SpriteList()
        self.player_bullet_list = arcade.SpriteList()
        self.enemy_bullet_list = arcade.SpriteList()
        self.shield_list = arcade.SpriteList(is_static=True)

        # Set up the player
        self.score = 0
        self.vida = 3

        self.player_sprite = arcade.Sprite("InvadoresDelEspacio\imagenes\A1.png", SPRITE_SCALING_PLAYER)
        self.player_sprite.center_x = 50
        self.player_sprite.center_y = 40
        self.player_list.append(self.player_sprite)

        # Make each of the shields
        for x in range(75, 800, 190):
            self.make_shield(x)

        # Set the background color
        arcade.set_background_color(arcade.color.AMAZON)

        self.setup_level_one()

    def on_draw(self):
        """ Render the screen. """

        # This command has to happen before we start drawing
        # self.clear()
        arcade.start_render() 
        arcade.draw_texture_rectangle(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2,SCREEN_WIDTH, SCREEN_HEIGHT, self.background)
        # Draw all the sprites.
        self.enemy_list.draw()
        self.player_bullet_list.draw()
        self.enemy_bullet_list.draw()
        self.shield_list.draw()
        self.player_list.draw()

        # Render the text
        arcade.draw_text(f"Score: {self.score}", 10, 20, arcade.color.WHITE, 14)

        if self.game_state == GAME_OVER:
            arcade.draw_text("GAME OVER", SCREEN_WIDTH/2, 300, arcade.color.WHITE, 55)
            self.set_mouse_visible(True)

        if self.game_state == WIN:
            arcade.draw_text("GANASTE", SCREEN_WIDTH/2, 300, arcade.color.WHITE, 55)
            self.set_mouse_visible(True)


    def on_key_press(self, key, modifiers):

        if self.game_state == GAME_OVER:
            return

        if key == arcade.key.UP:
            self.player_sprite.center_y  = self.player_sprite.center_y + PLAYER_SPEED
        elif key == arcade.key.DOWN:
            self.player_sprite.center_y  = self.player_sprite.center_y - PLAYER_SPEED
        elif key == arcade.key.RIGHT:
            self.player_sprite.center_x  = self.player_sprite.center_x + PLAYER_SPEED
        elif key == arcade.key.LEFT:
            self.player_sprite.center_x  = self.player_sprite.center_x - PLAYER_SPEED
            
  
        
        if key == arcade.key.SPACE:
            # Gunshot sound
            arcade.play_sound(self.gun_sound)
            # Create a bullet
            bullet = arcade.Sprite(":resources:images/space_shooter/laserBlue01.png", SPRITE_SCALING_LASER)
            # bullet2 = arcade.Sprite(":resources:images/space_shooter/laserBlue01.png", SPRITE_SCALING_LASER)
            # The image points to the right, and we want it to point up. So
            # rotate it.
            bullet.angle = 90
            # bullet2.angle = 90
            # Give the bullet a speed
            bullet.change_y = BULLER_PLAYER_SPEED
            # bullet2.change_y = BULLET_SPEED
            # Position the bullet
            bullet.center_x = self.player_sprite.center_x - 20
            bullet.bottom = self.player_sprite.top 
            # bullet2.center_x = self.player_sprite.center_x + 20
            # bullet2.bottom = self.player_sprite.top
            # Add the bullet to the appropriate lists
            self.player_bullet_list.append(bullet)
            # self.player_bullet_list.append(bullet2)

    def on_key_release(self, key, modifiers):
     if key == arcade.key.LEFT:
       self.player_sprite.change_x = 0
     elif key == arcade.key.RIGHT:
       self.player_sprite.change_x = 0

    def update_enemies(self):

        # Move the enemy vertically
        for enemy in self.enemy_list:
            enemy.center_x += self.enemy_change_x

        # Check every enemy to see if any hit the edge. If so, reverse the
        # direction and flag to move down.
        move_down = False
        for enemy in self.enemy_list:
            if enemy.right > RIGHT_ENEMY_BORDER and self.enemy_change_x > 0:
                self.enemy_change_x *= -1
                move_down = True
            if enemy.left < LEFT_ENEMY_BORDER and self.enemy_change_x < 0:
                self.enemy_change_x *= -1
                move_down = True

        # Did we hit the edge above, and need to move t he enemy down?
        if move_down:
            # Yes
            for enemy in self.enemy_list:
                # Move enemy down
                enemy.center_y -= ENEMY_MOVE_DOWN_AMOUNT
                # Flip texture on enemy so it faces the other way
                if self.enemy_change_x > 0:
                    enemy.texture = self.enemy_textures[0]
                else:
                    enemy.texture = self.enemy_textures[1]

    def allow_enemies_to_fire(self):

        x_spawn = []
        for enemy in self.enemy_list:
            # Adjust the chance depending on the number of enemies. Fewer
            # enemies, more likely to fire.
            chance = 1 + len(self.enemy_list) * 30

            # Fire if we roll a zero, and no one else in this column has had
            # a chance to fire.
            if random.randrange(chance) == 0 and enemy.center_x not in x_spawn:
                # Create a bullet
                bullet = arcade.Sprite(":resources:images/space_shooter/laserRed01.png", SPRITE_SCALING_LASER)

                # Angle down.
                bullet.angle = 180

                # Give the bullet a speed
                bullet.change_y = -BULLET_SPEED

                # Position the bullet so its top id right below the enemy
                bullet.center_x = enemy.center_x
                bullet.top = enemy.bottom

                # Add the bullet to the appropriate list
                self.enemy_bullet_list.append(bullet)

            # Ok, this column has had a chance to fire. Add to list so we don't
            # try it again this frame.
            x_spawn.append(enemy.center_x)

    def process_enemy_bullets(self):

        # Move the bullets
        self.enemy_bullet_list.update()

        # Loop through each bullet
        for bullet in self.enemy_bullet_list:
            # Check this bullet to see if it hit a shield
            hit_list = arcade.check_for_collision_with_list(bullet, self.shield_list)

            # If it did, get rid of the bullet and shield blocks
            if len(hit_list) > 0:
                bullet.remove_from_sprite_lists()
                for shield in hit_list:
                    shield.remove_from_sprite_lists()
                continue

            # See if the player got hit with a bullet
            if arcade.check_for_collision_with_list(self.player_sprite, self.enemy_bullet_list):
                self.game_state = GAME_OVER

            # If the bullet falls off the screen get rid of it
            if bullet.top < 0:
                bullet.remove_from_sprite_lists()


    def process_player_bullets(self):

        # Move the bullets
        self.player_bullet_list.update()

        # Loop through each bullet
        for bullet in self.player_bullet_list:

            # Check this bullet to see if it hit a enemy
            hit_list = arcade.check_for_collision_with_list(bullet, self.shield_list)
            # If it did, get rid of the bullet
            if len(hit_list) > 0:
                bullet.remove_from_sprite_lists()
                for shield in hit_list:
                    shield.remove_from_sprite_lists()
                continue

            # Check this bullet to see if it hit a enemy
            hit_list = arcade.check_for_collision_with_list(bullet, self.enemy_list)

            # If it did, get rid of the bullet
            if len(hit_list) > 0:
                bullet.remove_from_sprite_lists()

            # For every enemy we hit, add to the score and remove the enemy
            for enemy in hit_list:
                enemy.remove_from_sprite_lists()
                self.score += 1

                # Hit Sound
                arcade.play_sound(self.hit_sound)

            # If the bullet flies off-screen, remove it.
            if bullet.bottom > SCREEN_HEIGHT:
                bullet.remove_from_sprite_lists()

    def on_update(self, delta_time):
        """ Movement and game logic """
        # WordRecognizer()
        if self.game_state == GAME_OVER:
            return
        if self.game_state == WIN:
            return

        self.update_enemies()
        self.allow_enemies_to_fire()
        self.process_enemy_bullets()
        self.process_player_bullets()

        if len(self.enemy_list) == 0:
            self.game_state = WIN


def main():
    window = MyGame()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()