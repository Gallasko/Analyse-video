from threading import Thread

import cv2
import time

class VideoStream():
	"""docstring for VideoStrem"""
	def __init__(self, src=0):
		self.stream = cv2.VideoCapture(src)
		self.videoQueue = []
		self.grabbed, self.frame = self.stream.read()
		self.nbFrameRead = 0
		
		self.videoQueue.append(self.frame)

		self.fps = self.stream.get(cv2.CAP_PROP_FPS)

		self.stopped = False

		self.running = False
		self.started = False
		self.paused = True
		self.restartRequired = False

	def start(self):
		Thread(target=self.update, args=()).start()
		self.started = True
		self.running = True

	def update(self):
		while True:
			if self.stopped or not self.stream.isOpened():
				self.started = False
				return

			if self.running:
				self.paused = False
				self.grabbed, self.frame = self.stream.read()

				if self.grabbed:
					self.videoQueue.append(self.frame)
				self.paused = True

			if self.restartRequired:
				self.running = True
				self.restartRequired = False

	def readyRead(self):
		return len(self.videoQueue) > 0

	def read(self):
		self.nbFrameRead += 1
		return self.videoQueue.pop(0)

	def frameReaded(self):
		return self.nbFrameRead

	def getFps(self):
		return self.fps

	def pause(self):
		self.running = False

	def resume(self):
		if self.started == False:
			self.start()
		self.restartRequired = True

	def isPlaying(self):
		return self.running

	def getNbFrame(self):
		return self.stream.get(cv2.CAP_PROP_FRAME_COUNT)

	def setPos(self, pos):
		self.pause()
		while self.running:
			time.sleep(1)

		self.stream.set(cv2.CAP_PROP_POS_FRAMES, pos)
		self.nbFrameRead = pos
		self.videoQueue.clear()

		self.grabbed, self.frame = self.stream.read()
		if self.grabbed:
			self.videoQueue.append(self.frame)

	def stop(self):
		self.stopped = True
		self.stream.release()
		cv2.destroyAllWindows()

		