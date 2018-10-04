from enum import Enum 

class Move(Enum):
	FRONT = 0
	BACK = 1
	LEFT = 2
	RIGHT = 3
	UP = 4
	DOWN = 5


class Sticker(Enum):
	RED = 0
	GREEN = 1
	BLUE = 2
	YELLOW = 3
	ORANGE = 4
	WHITE = 5

class Cube:
	def __init__(self, size = 2):
		self.size = size
		self.fillCube()

	def fillCube(self):
		self.cube = [[[color for _ in range(self.size)] for _ in range(self.size)] for color in Sticker]