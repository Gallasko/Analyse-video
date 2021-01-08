import cv2

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui

class ResultWindow(QMainWindow): # Helper class to quickly print result to screen
	#Event#
	def resizeEvent(self, event):
		QMainWindow.resizeEvent(self, event)

		self.resize()

	def __init__(self, title, image):
		super(ResultWindow, self).__init__()
		self.title = title
		self.width = 640
		self.height = 480

		self.image = image

		self.initUI()

	def initUI(self):
		self.setGeometry(0,0, self.width, self.height)
		self.setWindowTitle(self.title)

		if self.image is not None:
			self.label = QLabel()
			self.setCentralWidget(self.label)

			self.setImage(self.image)

		self.show()

	def setImage(self, image):
		self.label.setPixmap(image)
		self.image = image

	def setHistogramme(self, hist):
		self.setCustomWidget(hist)

	def setCustomWidget(self, widget):
		self.setCentralWidget(widget)

	def resize(self):
		self.width = self.size().width()
		self.height = self.size().height()
