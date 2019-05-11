from itertools import cycle
import random
import sys
import math
import numpy as np
import time
from PIL import Image

import pygame
from pygame.locals import *


class Env(object):
    def __init__(self,fps=120):
        self.history_length=4
        self.action_space=[0,1]
        self.action_size = len(self.action_space)
        self.best_score=0
        self.get_score=0
        self.screen_width=84
        self.screen_height=84

        self.FPS = fps
        self.SCREENWIDTH = 288
        self.SCREENHEIGHT = 512
        self.PIPEGAPSIZE = 100
        self.BASEY = self.SCREENHEIGHT * 0.79
        self.IMAGES, self.SOUNDS, self.HITMASKS = {}, {}, {}
        self.PLAYERS_LIST = ((
                                 'assets/sprites/frog-upflap.png',
                                 'assets/sprites/frog-midflap.png',
                                 'assets/sprites/frog-downflap.png',
                             ),
        )
        self.BACKGROUNDS_LIST = (
            'assets/sprites/background.jpg',
            #'assets/sprites/background-day.png',
            #'assets/sprites/background-night.png',
        )
        self.PIPES_LIST = (
            'assets/sprites/pipe-green.png',
            #'assets/sprites/pipe-red.png',
        )
        pygame.init()
        self.FPSLOCK = pygame.time.Clock()
        self.SCREEN = pygame.display.set_mode((self.SCREENWIDTH, self.SCREENHEIGHT))
        pygame.display.set_caption('Flappy Frog')
        self.IMAGES['numbers'] = (
            pygame.image.load('assets/sprites/0.png').convert_alpha(),
            pygame.image.load('assets/sprites/1.png').convert_alpha(),
            pygame.image.load('assets/sprites/2.png').convert_alpha(),
            pygame.image.load('assets/sprites/3.png').convert_alpha(),
            pygame.image.load('assets/sprites/4.png').convert_alpha(),
            pygame.image.load('assets/sprites/5.png').convert_alpha(),
            pygame.image.load('assets/sprites/6.png').convert_alpha(),
            pygame.image.load('assets/sprites/7.png').convert_alpha(),
            pygame.image.load('assets/sprites/8.png').convert_alpha(),
            pygame.image.load('assets/sprites/9.png').convert_alpha()
        )
        self.IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
        self.IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
        self.IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

        randBg = random.randint(0, len(self.BACKGROUNDS_LIST) - 1)
        self.IMAGES['background'] = pygame.image.load(self.BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(self.PLAYERS_LIST) - 1)
        self.IMAGES['player'] = (
            pygame.image.load(self.PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(self.PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(self.PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(self.PIPES_LIST) - 1)
        self.IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(self.PIPES_LIST[pipeindex]).convert_alpha(), 180),
            pygame.image.load(self.PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        self.HITMASKS['pipe'] = (
            self.getHitmask(self.IMAGES['pipe'][0]),
            self.getHitmask(self.IMAGES['pipe'][1]),
        )

        # hitmask for player
        self.HITMASKS['player'] = (
            self.getHitmask(self.IMAGES['player'][0]),
            self.getHitmask(self.IMAGES['player'][1]),
            self.getHitmask(self.IMAGES['player'][2]),
        )
        self.reset()


    def mainGame(self,action):
        if action:
            if self.playery > -2 * self.IMAGES['player'][0].get_height():
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

        crashTest = self.checkCrash({'x': self.playerx, 'y': self.playery, 'index': self.playerIndex},
                                    self.upperPipes, self.lowerPipes)
        if crashTest[0]:
            return self.score
        # check for score
        playerMidPos = self.playerx + self.IMAGES['player'][0].get_width() / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + self.IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                self.get_score=1

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(self.playerIndexGen)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        playerHeight = self.IMAGES['player'][self.playerIndex].get_height()
        self.playery += min(self.playerVelY, self.BASEY - self.playery - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            self.newPipe = self.getRandomPipe()
            self.upperPipes.append(self.newPipe[0])
            self.lowerPipes.append(self.newPipe[1])

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -self.IMAGES['pipe'][0].get_width():
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # draw sprites
        self.SCREEN.blit(self.IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            self.SCREEN.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            self.SCREEN.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        self.SCREEN.blit(self.IMAGES['base'], (self.basex, self.BASEY))
        # print score so player overlaps the score
        self.showScore(self.score)
        self.SCREEN.blit(self.IMAGES['player'][self.playerIndex], (self.playerx, self.playery))

        pygame.display.update()
        self.FPSLOCK.tick(self.FPS)

    def reset(self):
        self.score = 0
        self.playerIndex = 0
        self.loopIter = 0
        self.playerIndexGen = cycle([0, 1, 2, 1])
        self.playerx, self.playery = int(self.SCREENWIDTH * 0.2), int(
            (self.SCREENHEIGHT - self.IMAGES['player'][0].get_height()) / 2)
        self.basex = 0
        self.baseShift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()

        self.pipeVelX = -4
        self.playerVelY = -9  # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10  # max vel along Y, max descend speed
        self.playerMinVelY = -8  # min vel along Y, max ascend speed
        self.playerAccY = 1  # players downward accleration
        self.playerFlapAcc = -9  # players speed on flapping
        self.playerFlapped = False  # True when player flaps

        self.newPipe1 = self.getRandomPipe()
        self.newPipe2 = self.getRandomPipe()
        self.upperPipes = [
            {'x': self.SCREENWIDTH + 0, 'y': self.newPipe1[0]['y']},
            {'x': self.SCREENWIDTH + 0 + (self.SCREENWIDTH / 2), 'y': self.newPipe2[0]['y']},
        ]

        # list of lowerpipe
        self.lowerPipes = [
            {'x': self.SCREENWIDTH + 0, 'y': self.newPipe1[1]['y']},
            {'x': self.SCREENWIDTH + 0 + (self.SCREENWIDTH / 2), 'y': self.newPipe2[1]['y']},
        ]
        return self.step(0)

    def step(self,action):
        screens=np.empty([self.history_length,self.screen_height,self.screen_width],dtype=np.float16)
        crashInfo = self.mainGame(action)
        screens[0][:][:] = self.observe(self.screen_height,self.screen_width)        
        for i in range(self.history_length-1):
            crashInfo = self.mainGame(0)
            screens[i+1][:][:]=self.observe(self.screen_height,self.screen_width)
        if crashInfo==None:
            if self.get_score==1:
                self.get_score=0
                return np.transpose(screens,(1,2,0)),1,action,0
            else:
                return np.transpose(screens,(1,2,0)),0,action,0
        else:
            if self.score>self.best_score:
                self.best_score=self.score
            print("Best score is :",self.best_score)
            
            return np.transpose(screens,(1,2,0)),-1,action,1

    def observe(self,height,width):
        img = pygame.surfarray.array3d(pygame.display.get_surface())
        img=Image.fromarray(img.astype('uint8')).convert('L')
        img=np.array(img.resize((height,width)))
        return img


    def playerShm(self,playerShm):
        """oscillates the value of playerShm['val'] between 8 and -8"""
        if abs(playerShm['val']) == 8:
            playerShm['dir'] *= -1

        if playerShm['dir'] == 1:
            playerShm['val'] += 1
        else:
            playerShm['val'] -= 1


    def getRandomPipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gapY = random.randrange(0, int(self.BASEY * 0.6 - self.PIPEGAPSIZE))
        gapY += int(self.BASEY * 0.2)
        pipeHeight = self.IMAGES['pipe'][0].get_height()
        pipeX = self.SCREENWIDTH + 10

        return [
            {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
            {'x': pipeX, 'y': gapY + self.PIPEGAPSIZE},  # lower pipe
        ]


    def showScore(self,score):
        """displays score in center of screen"""
        scoreDigits = [int(x) for x in list(str(score))]
        totalWidth = 0  # total width of all numbers to be printed

        for digit in scoreDigits:
            totalWidth += self.IMAGES['numbers'][digit].get_width()

        Xoffset = (self.SCREENWIDTH - totalWidth) / 2

        for digit in scoreDigits:
            self.SCREEN.blit(self.IMAGES['numbers'][digit], (Xoffset, self.SCREENHEIGHT * 0.1))
            Xoffset += self.IMAGES['numbers'][digit].get_width()


    def checkCrash(self,player, upperPipes, lowerPipes):
        """returns True if player collders with base or pipes."""
        pi = player['index']
        player['w'] = self.IMAGES['player'][0].get_width()
        player['h'] = self.IMAGES['player'][0].get_height()

        # if player crashes into ground
        if player['y'] + player['h'] >= self.BASEY - 1:
            return [True, True]
        else:

            playerRect = pygame.Rect(player['x'], player['y'],
                                     player['w'], player['h'])
            pipeW = self.IMAGES['pipe'][0].get_width()
            pipeH = self.IMAGES['pipe'][0].get_height()

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

                # player and upper/lower pipe hitmasks
                pHitMask = self.HITMASKS['player'][pi]
                uHitmask = self.HITMASKS['pipe'][0]
                lHitmask = self.HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = self.pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = self.pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    return [True, False]

        return [False, False]


    def pixelCollision(self,rect1, rect2, hitmask1, hitmask2):
        """Checks if two objects collide and not just their rects"""
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in range(rect.width):
            for y in range(rect.height):
                if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                    return True
        return False


    def getHitmask(self,image):
        """returns a hitmask using an image's alpha."""
        mask = []
        for x in range(image.get_width()):
            mask.append([])
            for y in range(image.get_height()):
                mask[x].append(bool(image.get_at((x, y))[3]))
        return mask
