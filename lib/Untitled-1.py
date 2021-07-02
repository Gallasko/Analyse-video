from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui

from .imagepipeline import ImagePipeline

import cv2
import time
import numpy as np
import threading

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

class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=-1):
        super(FlowLayout, self).__init__(parent)

        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)

        self.setSpacing(spacing)

        self.itemList = []

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self.itemList.append(item)

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        if index >= 0 and index < len(self.itemList):
            return self.itemList[index]

        return None

    def takeAt(self, index):
        if index >= 0 and index < len(self.itemList):
            return self.itemList.pop(index)

        return None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self.doLayout(QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()

        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())

        margin, _, _, _ = self.getContentsMargins()

        size += QSize(2 * margin, 2 * margin)
        return size

    def doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0

        for item in self.itemList:
            wid = item.widget()
            spaceX = self.spacing() + wid.style().layoutSpacing(QSizePolicy.PushButton, QSizePolicy.PushButton, Qt.Horizontal)
            spaceY = self.spacing() + wid.style().layoutSpacing(QSizePolicy.PushButton, QSizePolicy.PushButton, Qt.Vertical)
            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > rect.right() and lineHeight > 0:
                x = rect.x()
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y()

class CustomButtom(QLabel):
    sendText = pyqtSignal(str)

    def mousePressEvent(self, event):
        self.sendText.emit(self.text())

        if self.toggle:
            self.toggle = False
        else:
            self.toggle = True

    def enterEvent(self, event):
        self.setStyleSheet(cremeStyle)

    def leaveEvent(self, event):
        if self.toggle == False:
            self.setStyleSheet(whiteStyle)

    #Equipement Widget definition#
    def __init__(self, text):
        super(CustomButtom, self).__init__()
        self.setStyleSheet(whiteStyle)
        self.setText(text)

        self.toggle = False

class OutfitLabel(QWidget):
    sendInfo = pyqtSignal(str, int, np.ndarray)

    def mousePressEvent(self, event):
        self.sendInfo.emit(self.text(), self.frameTime, self.frame)

    def enterEvent(self, event):
        self.setStyleSheet(cremeStyle)

    def leaveEvent(self, event):
        self.setStyleSheet(whiteStyle)

    #Equipement Widget definition#
    def __init__(self, text, frameTime, frame, colors):
        super(OutfitLabel, self).__init__()
        self.setStyleSheet(whiteStyle)

        sizepolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSizePolicy(sizepolicy)

        self.frameTime = frameTime
        self.frame = frame
        self.colors = colors

        self.label = QLabel(self)
        self.colorPrinter = []

        hbox = QHBoxLayout()

        self.setText(text)
        hbox.addWidget(self.label)

        for x in range(len(colors)):
            print(colors[x])
            if not (colors[x][0] == 0 and colors[x][1] == 0 and colors[x][2] == 0):
                label = QLabel(self)

                pixmap = QPixmap(50, 50)
                pixmap.fill(QColor(255 * colors[x][0], 255 * colors[x][1], 255 * colors[x][2]))

                label.setPixmap(pixmap)
                self.colorPrinter.append(label)
                hbox.addWidget(label)

        hbox.setContentsMargins(5, 5, 5, 5)
        self.setLayout(hbox)

        self.width = self.label.width() + 10 + len(self.colorPrinter) * 55
        self.height = 60

        self.setMinimumSize(self.width, self.height)

    @pyqtSlot(list)
    def colorSearch(self, colorList):
        
        showed = False
        for color in self.colors:
            print(color)
            print((color[0] * 255, color[1] * 255, color[2] * 255))
            
            correspondingColor = True
            for x in range(3):
                if(colorList[x] > color[x] * 255 + 10 or colorList[x] < color[x] * 255 - 10):
                    correspondingColor = False

            if correspondingColor == True:
                self.show()
                showed = True

        if showed == False:
            self.hide()

    def setText(self, text):
        self.label.setText(text)

    def text(self):
        return self.label.text()

class ResultWidget(QWidget):
    showResult = pyqtSignal(int, list)

    def mousePressEvent(self, event):
        self.showResult.emit(self.frameTime, self.imageList)

    def enterEvent(self, event):
        self.setStyleSheet(cremeStyle)

    def leaveEvent(self, event):
        self.setStyleSheet(whiteStyle)

    def __init__(self, frameNumber, frameTime):
        super(ResultWidget, self).__init__()

        self.frameNumber = frameNumber
        self.frameTime = frameTime

        self.imageList = [] 
        self.labelList = []
        self.searchList = []

        self.initUI()

    def initUI(self):
        sizepolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSizePolicy(sizepolicy)

        self.fillWidget = QWidget(self)
        frameTimeSecond = int(self.frameTime % 60)
        frameTimeMinute = int(self.frameTime / 60)
        frameNumberLabel = QLabel("Frame number: {}".format(self.frameNumber))
        frameTimeLabel = QLabel("Frame time: {}:{}".format(frameTimeMinute, frameTimeSecond))

        hBox = QHBoxLayout()
        hBox.addWidget(frameNumberLabel)
        hBox.addWidget(frameTimeLabel)

        self.flowLayout = FlowLayout()

        layout = QVBoxLayout()
        layout.addLayout(hBox)
        layout.addLayout(self.flowLayout)

        self.setMinimumSize(250, len(self.imageList) * 70 + 40)

        self.fillWidget.setLayout(layout)

    def addWidget(self, widget, image, label):
        self.imageList.append(image)
        self.flowLayout.addWidget(widget)

        if label not in self.labelList:
            self.labelList.append(label)

        self.setMinimumSize(250, len(self.imageList) * 70 + 40)

    def getLabelList(self):
        return self.labelList

    @pyqtSlot(str, bool)
    def onSearchWidget(self, searchString, status):
        if status == True:
            self.searchList.append(searchString)
        else:
            self.searchList.remove(searchString)

        if self.searchList == []:
            self.show()
        else:
            res = all(elem in self.labelList for elem in self.searchList)

            if res:
                self.show()
            else:
                self.hide()
    
    def appendInSearchList(self, key):
        self.searchList.append(key)

class OutfitLabel(QWidget):
    sendInfo = pyqtSignal(str, int, np.ndarray, np.ndarray)

    def mousePressEvent(self, event):
        self.sendInfo.emit(self.text(), self.frameTime, self.frame, self.box)

    def enterEvent(self, event):
        self.setStyleSheet(cremeStyle)

    def leaveEvent(self, event):
        self.setStyleSheet(whiteStyle)

    #Equipement Widget definition#
    def __init__(self, text, frameTime, frame, colors, box):
        super(OutfitLabel, self).__init__()
        self.setStyleSheet(whiteStyle)

        sizepolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSizePolicy(sizepolicy)

        self.frameTime = frameTime
        self.frame = frame
        self.colors = colors
        self.box = box

        self.label = QLabel(self)
        self.colorPrinter = []

        hbox = QHBoxLayout()

        self.setText(text)
        hbox.addWidget(self.label)

        for x in range(len(colors)):
            print(colors[x])
            if not (colors[x][0] == 0 and colors[x][1] == 0 and colors[x][2] == 0):
                label = QLabel(self)

                pixmap = QPixmap(50, 50)
                pixmap.fill(QColor(255 * colors[x][0], 255 * colors[x][1], 255 * colors[x][2]))

                label.setPixmap(pixmap)
                self.colorPrinter.append(label)
                hbox.addWidget(label)

        hbox.setContentsMargins(5, 5, 5, 5)
        self.setLayout(hbox)

        self.width = self.label.width() + 10 + len(self.colorPrinter) * 55
        self.height = 60

        self.setMinimumSize(self.width, self.height)

            @pyqtSlot(list)
    def onCreateResultWidget(self, widgetInfo):
        widgetInit = widgetInfo[0]
        resultWidget = ResultWidget(widgetInit[0], widgetInit[1])

        for key in self.searchDict:
            if self.searchDict[key]:
                resultWidget.appendInSearchList(key)

        for x in range(len(widgetInfo)):
            if x == 0:
                pass
            else:
                if widgetInfo[x][0] not in self.searchList:
                    self.searchList.append(widgetInfo[x][0])
                    button = CustomButtom(widgetInfo[x][0])
                    button.sendText.connect(self.onSearchLabelClicked)
                    self.searchLayout.addWidget(button)
                    self.searchDict[widgetInfo[x][0]] = False

                label = OutfitLabel(widgetInfo[x][0], widgetInit[0], widgetInfo[x][1], widgetInfo[x][2], widgetInfo[x][3])
                label.sendInfo.connect(self.onOutfitLabelClicked)
                resultWidget.addWidget(label, widgetInfo[x][1], widgetInfo[x][0])

        self.resultHolder.appendWidget(resultWidget)
        #self.resultHolder.appendItem(QSpacerItem(100, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        resultWidget.showResult.connect(self.onWidgetClicked)
        self.searchWidget.connect(resultWidget.onSearchWidget)


class ResultHolder(QScrollArea):
    """docstring for ResultHolder"""

    #Equipement Holder definition#
    def __init__(self):
        super(ResultHolder, self).__init__()
        self.initUI()

    def initUI(self):
        sizepolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSizePolicy(sizepolicy)	
        
        self.setWidgetResizable(True)

        self.setBackgroundRole(QPalette.Light)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.mainWidget = QWidget()
        self.setWidget(self.mainWidget)

        self.layout = QVBoxLayout()
        self.mainWidget.setLayout(self.layout)

    def appendWidget(self, widget):
        self.layout.addWidget(widget)

    def appendItem(self, item):
        self.layout.addItem(item)

    def clear(self):
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

class FeatureWindow(QMainWindow): # Helper class to quickly print result to screen
    #Event#
    def resizeEvent(self, event):
        QMainWindow.resizeEvent(self, event)

        self.resize()

    #Signals#
    createResultWidget = pyqtSignal(list)
    setFrame = pyqtSignal(int)
    printFrame = pyqtSignal(int, np.ndarray)
    searchWidget = pyqtSignal(str, bool)
    searchColor = pyqtSignal(list)

    def __init__(self, title):
        super(FeatureWindow, self).__init__()
        self.title = title
        self.width = 640
        self.height = 480

        self.pipeline = ImagePipeline()
        self.running = False

        self.createResultWidget.connect(self.onCreateResultWidget)

        self.searchLayout = FlowLayout()
        self.searchList = []

        self.searchDict = {}

        self.initUI()

    def initUI(self):
        self.setGeometry(0,0, self.width, self.height)
        self.setWindowTitle(self.title)

        frameInputLabel = QLabel("Grab a frame each x seconds: ")
        self.frameInputEdit = QLineEdit()
        self.frameInputEdit.returnPressed.connect(self.startAnalyse)

        runButton = QPushButton("Run")
        runButton.clicked.connect(self.startAnalyse)

        hBox = QHBoxLayout()

        hBox.addWidget(frameInputLabel)
        hBox.addWidget(self.frameInputEdit)

        searchText = QLabel("Search: ")
        searchEdit = QLineEdit()
        #searchEdit.textChanged.connect()

        hBox2 = QHBoxLayout()
        hBox2.addWidget(searchText)
        hBox2.addWidget(searchEdit)

        colorPicker = QPushButton("Color")
        colorPicker.clicked.connect(self.colorPick)

        vBox = QVBoxLayout()

        vBox.addLayout(hBox)
        vBox.addWidget(runButton)
        vBox.addLayout(hBox2)
        vBox.addLayout(self.searchLayout)
        vBox.addWidget(colorPicker)
        vBox.addStretch()

        self.resultHolder = ResultHolder()

        self.mainWidget = QWidget()
        self.setCentralWidget(self.mainWidget)

        self.layout = QHBoxLayout()

        self.layout.addLayout(vBox)
        self.layout.addWidget(self.resultHolder)

        self.mainWidget.setLayout(self.layout)

    def showWidget(self):
        self.show()

    def setVideo(self, videoPath):
        self.videoPath = videoPath

    @pyqtSlot()
    def startAnalyse(self):
        if self.running == False:
            self.threadAnalyse = threading.Thread(target = self.analyse) # Lance le thread cycleThread
            self.threadAnalyse.start()

    @pyqtSlot()
    def colorPick(self):
        color = QColorDialog.getColor()
        colorList = [color.red(), color.green(), color.blue()]
        print(colorList)
        self.searchColor.emit(colorList)

    def stopAnalyse(self):
        self.running = False

    def analyse(self):
        print("Running the pipeline on a Video")
        fpsReading = int(self.frameInputEdit.text())

        self.running = True

        vidcap = cv2.VideoCapture(self.videoPath)
        count = 0
        success = True
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        frameCounted = 0

        timeElapsed = 0.0

        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        while success:
            success,image = vidcap.read()
            #print('read a new frame:',success)
            if count%(fps*int(fpsReading)) == 0 :
                t1 = time.perf_counter()
                print('successfully grabbed frame')
                frameCounted += 1

                result = self.pipeline.forward(image)

                resultWidget = None
                if(len(result[0]) != 0 or len(result[1]) != 0):
                    resultWidget = [(count, count/fps)]

                #Top Result
                for x in range(len(result[0])):
                    topBodyPrediction = [result[0][x]['prediction'][0][0], 0, result[0][x]['prediction'][0][2], result[0][x]['prediction'][0][3], result[0][x]['prediction'][0][4], 0, result[0][x]['prediction'][0][6]]

                    print("top")
                    resultWidget.append((class_names[np.argmax(topBodyPrediction)], result[0][x]['image'], result[0][x]['color']))

                #Bottom Result
                for x in range(len(result[1])):
                    print("bottom")

                    resultWidget.append(('Trouser', result[1][x]['image'], result[1][x]['color']))

                if resultWidget != None:
                    self.createResultWidget.emit(resultWidget)

                t2 = time.perf_counter()
                timeElapsed += t2 - t1

                print(f"Forwarding took: {t2 - t1:0.4f} seconds")
            count+=1
        
        print(f"Elapsed Time: {timeElapsed:0.2f}, Mean Time per image : {timeElapsed / frameCounted:0.4f}")

        self.running = False

    @pyqtSlot(list)
    def onCreateResultWidget(self, widgetInfo):
        widgetInit = widgetInfo[0]
        resultWidget = ResultWidget(widgetInit[0], widgetInit[1])

        for key in self.searchDict:
            if self.searchDict[key]:
                resultWidget.appendInSearchList(key)

        for x in range(len(widgetInfo)):
            if x == 0:
                pass
            else:
                if widgetInfo[x][0] not in self.searchList:
                    self.searchList.append(widgetInfo[x][0])
                    button = CustomButtom(widgetInfo[x][0])
                    button.sendText.connect(self.onSearchLabelClicked)
                    self.searchLayout.addWidget(button)
                    self.searchDict[widgetInfo[x][0]] = False

                label = OutfitLabel(widgetInfo[x][0], widgetInit[1], widgetInfo[x][1], widgetInfo[x][2])
                label.sendInfo.connect(self.onOutfitLabelClicked)
                self.searchColor.connect(label.colorSearch)
                resultWidget.addWidget(label, widgetInfo[x][1], widgetInfo[x][0])

        self.resultHolder.appendWidget(resultWidget)
        #self.resultHolder.appendItem(QSpacerItem(100, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        resultWidget.showResult.connect(self.onWidgetClicked)
        self.searchWidget.connect(resultWidget.onSearchWidget)

    @pyqtSlot(int, list)
    def onWidgetClicked(self, frameNumber):
        self.setFrame.emit(frameNumber)

    @pyqtSlot(str, int, np.ndarray)
    def onOutfitLabelClicked(self, string, frameNumber, frame):
        self.printFrame.emit(frameNumber, frame)

    @pyqtSlot(str)
    def onSearchLabelClicked(self, itemSearch):
        print(itemSearch)
        if self.searchDict[itemSearch] == False:
            self.searchDict[itemSearch] = True

            self.searchWidget.emit(itemSearch, True)
        else:
            self.searchDict[itemSearch] = False

            self.searchWidget.emit(itemSearch, False)

    def setCustomWidget(self, widget):
        self.setCentralWidget(widget)

    def resize(self):
        self.width = self.size().width()
        self.height = self.size().height()
