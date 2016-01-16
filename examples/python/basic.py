#!/usr/bin/python
from vizia import DoomGame
from random import choice

from time import sleep
from time import time

import cv2

game = DoomGame()
game.load_config("config_basic.properties")
game.init()

actions = [[True,False,False],[False,True,False],[False,False,True]]
left = actions[0]
right = actions[1]
shoot = actions[2]
idle = [False,False,False]

iters = 10000
sleep_time = 0.15

for i in range(iters):

	if game.is_episode_finished():
		
		print "episode finished!"
		print "summary reward:", game.get_summary_reward()
		print "************************"
		game.new_episode()

	s = game.get_state()
	r = game.make_action(choice(actions))
	
	print "state #" +str(s.number)
	print "ammo:", s.game_variables[0]
	print "reward:",r
	print "====================="	
	if sleep_time>0:
		sleep(sleep_time)
	


game.close()
