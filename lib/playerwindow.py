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
from .videostream import VideoStream

import cv2

import matplotlib
import matplotlib.pyplot as plt

from mrcnn import visualize

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
background-color: rgba(250, 125, 225, 60);
}
"""

#sliderStyle = """
#* {color: qlineargradient(spread:pad, x1:0 y1:0, x2:1 y2:0, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(255, 255, 255, 255));
#"""
#background: qlineargradient( x1:0 y1:0, x2:0.5 y2:0, x3:1 y3:0, stop:0 cyan, stop:1 blue, stop 2 red);}
#"""

sliderStyle = """
background: qlineargradient( x1:0 y1:0, x2:1 y2:0, stop:0 red, stop:0.05 red, stop:0.1 yellow, stop:0.15 green, stop:0.2 green, stop:0.3 green, stop:0.4 green, stop:0.55 green, stop:0.6 yellow, stop:0.65 red, stop:1 red);
"""

import numpy as np

def boolToColor(bool):
    if(bool):
        return "green"
    else:
        return "red"

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
        self.featureWindow.sendVisibleList.connect(self.onReceiveVisibleList)

        self.videoWidget = QVideoWidget()
        self.videoWidget.setStyleSheet(grayStyle)

        # appendus
        #self.stream = VideoStream(0)

        self.label = QLabel()

        #self.image = self.frameToPixmap(self.stream.read(), cv2.COLOR_BGR2RGB)

        #self.label.setPixmap(self.image)

        #self.setCentralWidget(self.label)

        #self.show()

        self.videoOverlay = QLabel(self.label)
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
        #layout.addWidget(self.videoWidget)

        self.gridLayout = QGridLayout()
        self.gridLayout.addWidget(self.label, 0, 0)
        self.gridLayout.addWidget(self.videoOverlay, 0, 0, Qt.AlignTop | Qt.AlignLeft)
        layout.addLayout(self.gridLayout)
        #layout.addWidget(self.label)

        self.currentImage = None

        layout.addLayout(controlLayout)
        layout.addWidget(self.errorLabel)

        # Set widget to contain window contents
        wid.setLayout(layout)

        #self.mediaPlayer.setVideoOutput(self.videoWidget)
        #self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        #self.mediaPlayer.positionChanged.connect(self.positionChanged)
        #self.mediaPlayer.durationChanged.connect(self.durationChanged)
        #self.mediaPlayer.error.connect(self.handleError)

    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie",
                QDir.homePath())

        print(QUrl.fromLocalFile(fileName))

        if fileName != '':
            self.stream = VideoStream(fileName)
            self.videoWidget.setStyleSheet(darkStyle)
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)

            self.featureWindow.setVideo(fileName)
            self.featureWindow.setFrame.connect(self.onSetFrame)
            self.featureWindow.showWidget()

            self.durationChanged(self.stream.getNbFrame())

    @pyqtSlot(int)
    def onSetFrame(self, frameTime):
        self.setPosition(frameTime)

    @pyqtSlot(list)
    def onReceiveVisibleList(self, visibleList):
        sliderStyle = ""

        stopLen = len(visibleList)

        print(visibleList)

        if stopLen > 0:
            sliderStyle = "background: qlineargradient(x1:0 y1:0, x2:1 y2:0"

            stopHelper = (1.0 / stopLen) / 3.0

            for i in range(stopLen):
                if i == 0: 
                    sliderStyle += ", stop: {} {}".format(0, boolToColor(visibleList[i]))
                    sliderStyle += ", stop: {} {}".format(0 + stopHelper, boolToColor(visibleList[i]))
                elif i < stopLen - 1:
                    currentStop = i * (1.0 / stopLen)
                    sliderStyle += ", stop: {} {}".format(currentStop - stopHelper, boolToColor(visibleList[i]))
                    sliderStyle += ", stop: {} {}".format(currentStop, boolToColor(visibleList[i]))
                    sliderStyle += ", stop: {} {}".format(currentStop + stopHelper, boolToColor(visibleList[i]))
                elif i == stopLen - 1:
                    sliderStyle += ", stop: {} {}".format(1 - stopHelper, boolToColor(visibleList[i]))
                    sliderStyle += ", stop: {} {}".format(1, boolToColor(visibleList[i]))
                    
            sliderStyle += ");"

        self.positionSlider.setStyleSheet(sliderStyle)

    @pyqtSlot(int, np.ndarray, np.ndarray)
    def onPrintFrame(self, frameTime, frame, box):
        print(frameTime)
        currentImage = self.setPosition(frameTime)
        currentImage = cv2.cvtColor(currentImage, cv2.COLOR_BGR2BGRA)

        cv2.imwrite('Temp.png', frame)
        img = cv2.imread('Temp.png')

        # read image
        #ht, wd, cc = img.shape

        # create new image of desired size and color (blue) for padding
        wd = box[2]
        ht = box[3]
        #color = (0,0,0)
        #result = np.full((hh,ww, cc), color, dtype=np.uint8)

        # compute center offset
        xx = box[0]
        yy = box[1]

        """
        tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
        b, g, r = cv2.split(img)
        rgba = [b,g,r, alpha]
        dst = cv2.merge(rgba,4)
        """

        # copy img image into center of result image
        #currentImage[yy:yy+ht, xx:xx+wd] = frame
        #currentImage[xx:xx+img.shape[0], yy:yy+img.shape[1]] = img
        self.rgb_weights = [0.2989, 0.5870, 0.1140]
        grayscale_image = np.dot(img[...,:3], self.rgb_weights)

        colors = visualize.random_colors(1)
        print(colors)

        alpha = 0.5

        for c in range(3):
            currentImage[:, :, c] = np.where(grayscale_image > 0.2, currentImage[:, :, c] * (1 - alpha) + alpha * colors[0][c] * 255, currentImage[:, :, c])

        #currentImage[0:img.shape[0], 0:img.shape[1]] = img
        #masked_image = currentImage.astype(np.uint32).copy()
        #colors =  visualize.random_colors(3)
        #masked_image = visualize.apply_mask(masked_image, frame, np.asarray(colors))
        #currentImage[xx:xx+wd, yy:yy+ht] = img

        
        self.label.setPixmap(self.frameToPixmap(currentImage, cv2.COLOR_BGR2RGB))


        #black_mask = np.all(result == 0, axis=-1)
        #alpha = np.uint8(np.logical_not(black_mask)) * 255
        #bgra = np.dstack((result, alpha))
        #cv2.imwrite("Temp.png", bgra)

        #image = QImage('Temp.jpg')
        #image = image.convertToFormat(QImage.Format_ARGB32)
        #pixmap = QPixmap('Temp.png')
        #pixmap.scaled(self.width, self.height, Qt.KeepAspectRatio) 

        
        #self.videoOverlay.setPixmap(pixmap)
        #styleSheet = "QWidget{ padding: " + str(box[0]) + " " + str(box[1]) + "0 0;}"
        #self.videoOverlay.setStyleSheet(styleSheet)
        #self.videoOverlay.move(box[0], box[1])
        #self.gridLayout.setHorizontalSpacing(box[0])
        #self.gridLayout.setVerticalSpacing(box[1])
        #self.videoOverlay.show()

        #if self.imageMask == None:
        #    self.imageMask = ResultWindow("Mask", pixmap)
        #else:
        #    self.imageMask.setImage(pixmap)

        plt.imshow(frame, cmap=plt.cm.binary)
        plt.show()

    def exitCall(self):
        sys.exit(app.exec_())

    def play(self):
        if self.stream.isPlaying():
            self.stream.pause()
        else:
            self.stream.resume()
            QTimer.singleShot(0, self.printImage)
<<<<<<< HEAD

        #self.videoOverlay.hide()

=======

        #self.videoOverlay.hide()

>>>>>>> 7c9476d3424e5e4e5be9591d4971f5f2c41e5290
    def printImage(self):
        if self.stream.isPlaying():
            if self.stream.readyRead():
                self.currentImage = self.stream.read()
                self.label.setPixmap(self.frameToPixmap(self.currentImage, cv2.COLOR_BGR2RGB))
                self.positionChanged(self.stream.frameReaded())
                QTimer.singleShot(self.stream.getFps(), self.printImage)
            else:
                QTimer.singleShot(0, self.printImage)

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

        #self.videoOverlay.hide()

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

        #self.videoOverlay.hide()

    def durationChanged(self, duration):
        print(duration)
        self.positionSlider.setRange(0, duration)

        #self.videoOverlay.hide()

    def setPosition(self, position):
        print(position)
        self.stream.setPos(position)
    
        while self.stream.readyRead() == False:
            time.sleep(0)

        self.currentImage = self.stream.read()
        self.label.setPixmap(self.frameToPixmap(self.currentImage, cv2.COLOR_BGR2RGB))

        return self.currentImage

        #self.videoOverlay.hide()

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