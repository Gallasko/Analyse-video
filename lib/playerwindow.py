# PyQt5 Video player
#!/usr/bin/env python

from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui

import sys

from .featurewindow import FeatureWindow
from .resultwindow import ResultWindow

import cv2

import matplotlib
import matplotlib.pyplot as plt

appStyle = """
QMainWindow{
background-color: #6c757d;
}
"""

grayStyle = """
QVideoWidget{
background-color: #343a40;
}
"""

darkStyle = """
QVideoWidget{
background-color: #212529;
}
"""

whiteStyle = """
QVideoWidget{
background-color: #FFFFFF;
}
"""

cremeStyle = """
QWidget{
background-color: #FFE6CD;
}
"""

import numpy as np

class PlayerWindow(QMainWindow):
    def __init__(self, parent=None):
        super(PlayerWindow, self).__init__(parent)
        self.width = 640
        self.height = 480

        matplotlib.use('QT5Agg')

        self.setGeometry(0,0, self.width, self.height)

        self.setWindowTitle("PyQt Video Player Widget") 
        self.setStyleSheet(appStyle)

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        self.featureWindow = FeatureWindow("List of Features")
        self.featureWindow.printFrame.connect(self.onPrintFrame)

        self.videoWidget = QVideoWidget()
        self.videoWidget.setStyleSheet(grayStyle)

        self.videoOverlay = QLabel(self.videoWidget)
        self.videoOverlay.setStyleSheet(cremeStyle)
        self.videoOverlay.show()

        self.imageMask = None

        layout_box = QHBoxLayout(self.videoWidget)
        layout_box.setContentsMargins(0, 0, 0, 0)
        layout_box.addWidget(self.videoOverlay)

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred,
                QSizePolicy.Maximum)

        # Create new action
        openAction = QAction(QIcon('open.png'), '&Open', self)        
        openAction.setShortcut('Ctrl+O')
        openAction.setStatusTip('Open movie')
        openAction.triggered.connect(self.openFile)

        # Create exit action
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)        
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.exitCall)

        # Create menu bar and add action
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')
        #fileMenu.addAction(newAction)
        fileMenu.addAction(openAction)
        fileMenu.addAction(exitAction)

        # Create a widget for window contents
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # Create layouts to place inside widget
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        layout = QVBoxLayout()
        layout.addWidget(self.videoWidget)
        layout.addLayout(controlLayout)
        layout.addWidget(self.errorLabel)

        # Set widget to contain window contents
        wid.setLayout(layout)

        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)

    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie",
                QDir.homePath())

        if fileName != '':
            self.videoWidget.setStyleSheet(darkStyle)
            self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)

            self.featureWindow.setVideo(fileName)
            self.featureWindow.setFrame.connect(self.onSetFrame)
            self.featureWindow.showWidget()

    @pyqtSlot(int)
    def onSetFrame(self, frameTime):
        self.setPosition(frameTime * 1000)

    @pyqtSlot(int, np.ndarray)
    def onPrintFrame(self, frameTime, frame):
        self.setPosition(frameTime * 1000)

        self.videoOverlay.setPixmap(self.frameToPixmap(frame, cv2.COLOR_BGR2GRAY))
        self.videoOverlay.show()

        if self.imageMask == None:
            self.imageMask = ResultWindow("Mask", self.frameToPixmap(frame, cv2.COLOR_BGR2GRAY))
        else:
            self.imageMask.setImage(self.frameToPixmap(frame, cv2.COLOR_BGR2GRAY))

        plt.imshow(frame, cmap=plt.cm.binary)
        plt.show()

    def exitCall(self):
        sys.exit(app.exec_())

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

        self.videoOverlay.hide()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

        self.videoOverlay.hide()

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

        self.videoOverlay.hide()

    def durationChanged(self, duration):
        print(duration)
        self.positionSlider.setRange(0, duration)

        self.videoOverlay.hide()

    def setPosition(self, position):
        print(position)
        self.mediaPlayer.setPosition(position)

        self.videoOverlay.hide()

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())

    def frameToPixmap(self, frame, color): #convert a opencv image to a qt pixmap 
        if color == cv2.COLOR_BGR2RGB:
            rgbImage = cv2.cvtColor(frame, color)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w

            convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
            p = convertToQtFormat.scaled(self.width, self.height, Qt.KeepAspectRatio)

        elif color == cv2.COLOR_BGR2GRAY:
            frame = np.asarray(frame, dtype = np.uint8)
            h, w = frame.shape[:2]

            convertToQtFormat = QtGui.QImage(frame.data, w, h, QImage.Format_Grayscale8) 
            p = convertToQtFormat.scaled(self.width, self.height, Qt.KeepAspectRatio)

        return QPixmap.fromImage(p)