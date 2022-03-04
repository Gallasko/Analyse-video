from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from skimage.io.collection import ImageCollection

from .imagepipeline import ImagePipeline
from .colorcircle import ColorCircleDialog
from .openglwidget2 import GLWidget

import os
import cv2
import time
import numpy as np
import math
import threading
import asyncio
import pickle

whiteStyle = """
QVideoWidget{
background-color: #FFFFFF;
}
"""

#whiteStyle = """
#* {color: qlineargradient(spread:pad, x1:0 y1:0, x2:1 y2:0, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(255, 255, 255, 255));
#background: qlineargradient( x1:0 y1:0, x2:1 y2:0, stop:0 cyan, stop:1 blue);}
#"""

cremeStyle = """
QWidget{
background-color: #FFE6CD;
}
"""

#TODO multiples personnes detection across 2 or 3 frames
#TODO pipette pour selectionner une couleur de la video

def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v

def rgb_to_l(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0

    if r <= 0.03928:
        r = r / 12.92
    else:
        r = ((r + 0.055) / 1.055) ** 2.4

    if g <= 0.03928:
        g = g / 12.92
    else:
        g = ((g + 0.055) / 1.055) ** 2.4

    if b <= 0.03928:
        b = b / 12.92
    else:
        b = ((b + 0.055) / 1.055) ** 2.4

    return (0.2126 * r + 0.7152 * g + 0.0722 * b)

def rgb_to_xyz(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0

    if r <= 0.03928:
        r = r / 12.92
    else:
        r = ((r + 0.055) / 1.055) ** 2.4

    if g <= 0.03928:
        g = g / 12.92
    else:
        g = ((g + 0.055) / 1.055) ** 2.4

    if b <= 0.03928:
        b = b / 12.92
    else:
        b = ((b + 0.055) / 1.055) ** 2.4

    r, g, b, = r * 100, g * 100, b * 100

    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    return x, y, z

def xyz_to_lab(x, y, z):
    REF_X = 95.047
    REF_Y = 100.000
    REF_Z = 108.883
    x, y, z = x/REF_X, y/REF_Y, z/REF_Z

    if x > 0.008856:
        x = x ** 0.3333333
    else:
        x = (7.787 * x) + (16.0 / 116.0)

    if y > 0.008856:
        y = y ** 0.3333333
    else:
        y = (7.787 * y) + (16.0 / 116.0)

    if z > 0.008856:
        z = z ** 0.3333333
    else:
        z = (7.787 * z) + (16.0 / 116.0)

    l = (116.0 * y) - 16.0
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    return l, a, b

def lab_to_lch(l, a, b):
    l = l
    c = math.sqrt((a * a) + (b * b))
    h = math.atan2(b, a)

    return l, c, h

def deltaColor(bgrTuple1, bgrTuple2):
    b1, g1, r1 = bgrTuple1
    b2, g2, r2 = bgrTuple2    

    x1, y1, z1 = rgb_to_xyz(r1, g1, b1)
    l1, a1, b1 = xyz_to_lab(x1, y1, z1)
    l1, c1, h1 = lab_to_lch(l1, a1, b1)

    x2, y2, z2 = rgb_to_xyz(r2, g2, b2)
    l2, a2, b2 = xyz_to_lab(x2, y2, z2)
    l2, c2, h2 = lab_to_lch(l2, a2, b2)

    avg_L = ( l1 + l2 ) * 0.5
    delta_L = l1 - l2
    avg_C = ( c1 + c2 ) * 0.5
    delta_C = c1 - c2
    avg_H = ( h1 + h2 ) * 0.5

    CV_PI = np.pi

    if( math.fabs( h1 - h2 ) > CV_PI ):
        avg_H += CV_PI

    delta_H = h1 - h2
    if( math.fabs( delta_H ) > CV_PI ):
        if( h1 <= h2 ):
            delta_H += CV_PI * 2.0
        else:
            delta_H -= CV_PI * 2.0

    delta_H = math.sqrt( c1 * c2 ) * math.sin( delta_H ) * 2.0
    T = 1.0 - \
            0.17 * math.cos( avg_H - CV_PI / 6.0 ) + \
            0.24 * math.cos( avg_H * 2.0 ) + \
            0.32 * math.cos( avg_H * 3.0 + CV_PI / 30.0 ) - \
            0.20 * math.cos( avg_H * 4.0 - CV_PI * 7.0 / 20.0 )
    SL = avg_L - 50.0
    SL *= SL
    SL = SL * 0.015 / math.sqrt( SL + 20.0 ) + 1.0
    SC = avg_C * 0.045 + 1.0
    SH = avg_C * T * 0.015 + 1.0
    delta_Theta = avg_H / 25.0 - CV_PI * 11.0 / 180.0
    delta_Theta = math.exp( delta_Theta * -delta_Theta ) * ( CV_PI / 6.0 )
    RT = math.pow( avg_C, 7.0 )
    RT = math.sqrt( RT / ( RT + 6103515625.0 ) ) * math.sin( delta_Theta ) * -2.0#; // 6103515625 = 25^7

    delta_L /= SL
    delta_C /= SC
    delta_H /= SH

    return math.sqrt( delta_L * delta_L + delta_C * delta_C + delta_H * delta_H + RT * delta_C * delta_H )

class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=-1):
        super(FlowLayout, self).__init__(parent)

        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)

        self.setSpacing(10)

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

    def getWidgetList(self):
        return self.itemList

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
    sendInfo = pyqtSignal(str, int, np.ndarray, np.ndarray)

    def mousePressEvent(self, event):
        self.sendInfo.emit(self.text(), self.frameTime, self.fullFrame, self.box)

    def enterEvent(self, event):
        self.setStyleSheet(cremeStyle)

    def leaveEvent(self, event):
        self.setStyleSheet(whiteStyle)

    #Equipement Widget definition#
    def __init__(self, text, frameTime, frameNumber, fullFrame, colors, box, frame):
        super(OutfitLabel, self).__init__()
        self.setStyleSheet(whiteStyle)

        #sizepolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        #self.setSizePolicy(sizepolicy)

        self.frameTime = frameTime
        self.frameNumber = frameNumber
        self.fullFrame = fullFrame
        self.colors = colors
        self.box = box
        self.frame = frame
        self.showed = True

        self.label = QLabel(self)
        self.colorPrinter = []

        vBox = QVBoxLayout()
        hBox = QHBoxLayout()

        self.setText(text)
        hBox.addWidget(self.label)

        for x in range(len(colors)):
            if not (colors[x][0] == 0 and colors[x][1] == 0 and colors[x][2] == 0):
                label = QLabel(self)

                pixmap = QPixmap(50, 50)
                pixmap.fill(QColor(255 * colors[x][2], 255 * colors[x][1], 255 * colors[x][0]))

                label.setPixmap(pixmap)
                self.colorPrinter.append(label)
                hBox.addWidget(label)

        hBox.setContentsMargins(5, 5, 5, 5)

        imageLabel = QLabel()
        imageLabel.setPixmap(self.frameToPixmap(self.frame, cv2.COLOR_BGR2RGB, 300, 200))

        vBox.addWidget(imageLabel)
        vBox.addLayout(hBox)
        self.setLayout(vBox)

        self.w = self.label.width() + 10 + len(self.colorPrinter) * 55
        self.h = 60

        self.setMinimumSize(self.w, self.h)

    def showLabel(self):
        self.setMinimumSize(self.w, self.h)
        self.showed = True
        self.show()

    def hideLabel(self):
        self.setMinimumSize(0, 0)
        self.showed = False
        self.hide()

    def getValues(self):
        return self.label.text(), self.frameTime, self.frameNumber, self.fullFrame, self.colors, self.box, self.frame

    def colorSearch(self, colorList1, colorList2):
        self.showed = False
        #l2 = rgb_to_l(colorList1[0], colorList1[1], colorList1[2])
        #l3 = rgb_to_l(colorList2[0], colorList2[1], colorList2[2])
        for color in self.colors:
            #print(color)

            #h, s, v = rgb_to_hsv(color[2] * 255, color[1] * 255, color[0] * 255)
            #l = rgb_to_l(color[2] * 255, color[1] * 255, color[0] * 255)
            
            correspondingColor = True

            deltaC = deltaColor((color[2] * 255, color[1] * 255, color[0] * 255), (colorList1[0], colorList1[1], colorList1[2]))
            
            if(colorList2 != [0, 0, 0]):
                deltaC2 = deltaColor((color[2] * 255, color[1] * 255, color[0] * 255), (colorList2[0], colorList2[1], colorList2[2]))
            
                if deltaC > 15.0 or deltaC2 > 15.0:
                    correspondingColor = False
            else:
                if deltaC > 15.0:
                    correspondingColor = False
            #if l < l2:
            #    contrastRatio = (l + 0.05) / (l2 + 0.05)
            #else:
            #    contrastRatio = (l2 + 0.05) / (l + 0.05)

            #print(l, l2, contrastRatio)
            #print((color[0] * 255, color[1] * 255, color[2] * 255))
            
            #if h < colorList[0] - 10 or h > colorList[0] + 10:

            if correspondingColor == True:
                self.setMinimumSize(self.w, self.h)
                self.show()
                self.showed = True

        if self.showed == False:
            self.setMinimumSize(0, 0)
            self.hide()

        return self.showed

    @pyqtSlot()
    def resetWidget(self):
        if self.showed == False:
            self.setMinimumSize(self.w, self.h)
            self.showed = True
            self.show()

    def setText(self, text):
        self.label.setText(text)

    def text(self):
        return self.label.text()

    def isVisible(self):
        if(self.showed):
            return True
        else:
            return False
        #return self.showed
        
    def frameToPixmap(self, frame, color, width, height): #convert a opencv image to a qt pixmap 
        if color == cv2.COLOR_BGR2RGB:
            rgbImage = cv2.cvtColor(frame, color)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w

            convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
            p = convertToQtFormat.scaled(width, height, Qt.KeepAspectRatio)

        elif color == cv2.COLOR_BGR2GRAY:
            frame = np.asarray(frame, dtype = np.uint8)
            h, w = frame.shape[:2]

            convertToQtFormat = QtGui.QImage(frame.data, w, h, QImage.Format_Grayscale8) 
            p = convertToQtFormat.scaled(width, height, Qt.KeepAspectRatio)

        return QPixmap.fromImage(p)

class ResultWidget(QWidget):
    showResult = pyqtSignal(int)
    sendVisibleList = pyqtSignal(int, bool)

    def mousePressEvent(self, _):
        print(self.renderList)
        self.showResult.emit(self.frameTime)

    def enterEvent(self, _):
        self.mainWidget.setStyleSheet(cremeStyle)

    def leaveEvent(self, _):
        self.mainWidget.setStyleSheet(whiteStyle)

    def __init__(self, frameTime, frameNumber):
        super(ResultWidget, self).__init__()

        self.frameNumber = frameNumber
        self.frameTime = frameTime

        self.imageList = [] 
        self.labelList = []
        self.searchList = []
        self.widgetList = []
        self.renderList = []

        self.widgetId = 0
        self.visible = True

        self.initUI()

    def initUI(self):
        sizepolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSizePolicy(sizepolicy)

        #self.fillWidget = QWidget(self)
        frameTimeSecond = int(self.frameNumber % 60)
        frameTimeMinute = int(self.frameNumber / 60)
        self.frameNumberLabel = QLabel("Frame number: {}".format(self.frameTime))
        self.frameTimeLabel = QLabel("Frame time: {}:{}".format(frameTimeMinute, frameTimeSecond))

        self.hBox = QHBoxLayout()
        self.hBox.addWidget(self.frameNumberLabel)
        self.hBox.addWidget(self.frameTimeLabel)

        #self.flowLayout = QVBoxLayout()
        self.flowLayout = QGridLayout()
        
        layout = QVBoxLayout()
        layout.addLayout(self.hBox)
        layout.addLayout(self.flowLayout)
        layout.addStretch()

        #print(self.nbVisibleWidget(), len(self.imageList))

        self.setMinimumSize(625, len(self.renderList) * 70 + 40)

        #if len(self.renderList) == 0:
        #    self.hide()

        #self.setMinimumSize(250, self.nbVisibleWidget() * 70 + 40)
        #self.setMaximumSize(250, self.nbVisibleWidget() * 70 + 40)

        #self.setLayout(layout)
        self.mainWidget = QWidget(self)
        self.mainWidget.setLayout(layout)

        #self.mainWidget.show()
        #self.fillWidget.setLayout(layout)

    def setWidgetId(self, id):
        self.widgetId = id

    def getWidgetId(self):
        return self.widgetId

    def getWidgetList(self):
        return self.widgetList

    def addWidget(self, widget, label):
        #self.imageList.append(image)
        length = len(self.widgetList)

        self.widgetList.append(widget)

        self.flowLayout.addWidget(widget, int(length / 2), int(length % 2))

        self.renderList.append(widget)
        #widget.changed.connect(self.onWidgetChanged)

        if label not in self.labelList:
            self.labelList.append(label)

        #print(self.nbVisibleWidget(), len(self.imageList))

        self.setMinimumSize(625, int((len(self.widgetList) + 1) / 2) * 300 + 40)

        #if len(self.widgetList) > 0:
        #    self.show()

        #self.setMinimumSize(250, self.nbVisibleWidget() * 70 + 40)
        #self.setMaximumSize(250, self.nbVisibleWidget() * 70 + 40)

    def getLabelList(self):
        return self.labelList

    def colorSearch(self, colorList1, colorList2):
        self.renderList = []

        for widget in self.widgetList:
            if widget.colorSearch(colorList1, colorList2):
                widget.showLabel()
                self.renderList.append(widget)
            else:
                widget.hideLabel()

        while self.flowLayout.count():
            self.flowLayout.takeAt(0)

        self.setMinimumSize(625, int((len(self.renderList) + 1) / 2) * 300 + 40)

        if len(self.renderList) == 0:
            self.hide()
        else:
            self.show()

        length = 0

        for widget in self.renderList:
            self.flowLayout.addWidget(widget, int(length / 2), int(length % 2))
            length += 1

        return (len(self.renderList) > 0 and self.visible)

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
                self.visible = True
                self.show()
            else:
                self.visible = False
                self.hide()
    
    @pyqtSlot()
    def onWidgetChanged(self):
        self.renderList = []

        visibleList = []

        for widget in self.widgetList:
            visibleList.append(widget.isVisible())

            if widget.isVisible():
                widget.showLabel()
                self.renderList.append(widget)
            else:
                widget.hideLabel()

        #del self.flowLayout
        #self.flowLayout = FlowLayout()
        #self.flowLayout.clear()
        #del self.flowLayout
        #self.flowLayout = QVBoxLayout()

        while self.flowLayout.count():
            self.flowLayout.takeAt(0)

        self.setMinimumSize(625, int((len(self.renderList) + 1) / 2) * 300 + 40)

        if len(self.renderList) == 0:
            self.hide()
        else:
            self.show()

        length = 0

        for widget in self.renderList:
            self.flowLayout.addWidget(widget, int(length / 2), int(length % 2))
            length += 1

        if(len(self.renderList) > 0 and self.visible):
            self.sendVisibleList.emit(self.widgetId, True)
        else:
            self.sendVisibleList.emit(self.widgetId, False)

        #layout = QVBoxLayout()
        #layout.addLayout(self.hBox)
        #if len(self.renderList) != 0:
        #    layout.addLayout(self.flowLayout)
        #fillWidget = QWidget()
        #fillWidget.setLayout(layout)
        ##self.setCentral fillWidget

        #self.setMinimumSize(250, self.nbVisibleWidget() * 70 + 40)

        #self.setMinimumSize(250, self.nbVisibleWidget() * 70 + 40)
        #self.setMaximumSize(250, self.nbVisibleWidget() * 70 + 40)

    def appendInSearchList(self, key):
        self.searchList.append(key)

    def nbOutfitRendered(self):
        return len(self.renderList)

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

class ResultHolder(QScrollArea):
    """docstring for ResultHolder"""
    sendVisibleList = pyqtSignal(list)

    #Equipement Holder definition#
    def __init__(self):
        super(ResultHolder, self).__init__()
        self.initUI()

    def initUI(self):
        sizepolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSizePolicy(sizepolicy)	
        
        self.setWidgetResizable(True)

        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.mainWidget = QWidget()
        self.setWidget(self.mainWidget)

        self.layout = QVBoxLayout()
        self.mainWidget.setLayout(self.layout)

        self.mainWidget.setBackgroundRole(QPalette.Light)

        self.nextWidgetId = 0
        self.visibleWidget = []

    def getResultLists(self):
        imageList = []
        metadataList = []

        for it in range(self.layout.count()):
            if self.layout.itemAt(it).widget() is not None:
                widgetList = self.layout.itemAt(it).widget().getWidgetList()

                for widget in widgetList:
                    text, frameTime, frameNumber, fullFrame, colors, box, frame = widget.getValues()

                    metadataList.append({'frame time': frameTime, 'frame number': frameNumber, 'text': text, 'color': colors, 'box': box})
                    
                    imageList.append(frame)
                    imageList.append(fullFrame)

        return imageList, metadataList

    @pyqtSlot(list, list)
    def colorSearch(self, colorList1, colorList2):
        for it in range(self.layout.count()):
            if self.layout.itemAt(it).widget() is not None:
                self.visibleWidget[self.layout.itemAt(it).widget().getWidgetId()] = self.layout.itemAt(it).widget().colorSearch(colorList1, colorList2)

        self.sendVisibleList.emit(self.visibleWidget)

    @pyqtSlot(int, bool)
    def onWidgetVisibilityChanged(self, id, visible):
        self.visibleWidget[id] = visible
        self.sendVisibleList.emit(self.visibleWidget)

    def appendWidget(self, widget):
        widget.setWidgetId(self.nextWidgetId)
        self.nextWidgetId += 1
        self.visibleWidget.append(True)
        self.layout.addWidget(widget)

    def appendEmpty(self):
        self.nextWidgetId += 1
        self.visibleWidget.append(False)

    def appendItem(self, item):
        self.layout.addItem(item)

    def clear(self):
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.nextWidgetId = 0

class ColorAvatarDialog(QDialog):
    avatarColor = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Select Avatar")

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        colorWheel = ColorCircleDialog()
        colorWheel.currentColorChanged.connect(self.onColorModified)

        self.avatarWidget = GLWidget(self, True)
        self.avatarColor.connect(self.avatarWidget.changeColor)

        self.layout = QVBoxLayout()
        message = QLabel("Choose Avatar: ")
        self.layout.addWidget(message)

        # Color wheel to be shown as a dialog when the user click on a cloth
        self.layout.addWidget(colorWheel)
        
        self.layout.addWidget(self.avatarWidget)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def getClothes(self):
        return self.avatarWidget.getSelectedClothes()

    @pyqtSlot(QColor)
    def onColorModified(self, color):
        #self.searchColor.emit([color.hue(), color.saturation(), color.value()])
        #self.searchColor.emit([color.red(), color.green(), color.blue()])
        self.avatarColor.emit([color.red() / 255, color.green() / 255, color.blue() / 255])

class FeatureWindow(QMainWindow): # Helper class to quickly print result to screen
    #Event#
    def resizeEvent(self, event):
        QMainWindow.resizeEvent(self, event)

        self.resize()

    #Signals#
    createResultWidget = pyqtSignal(list)
    createEmptyWidget = pyqtSignal()
    setFrame = pyqtSignal(int)
    sendVisibleList = pyqtSignal(list)
    printFrame = pyqtSignal(int, np.ndarray, np.ndarray)
    searchWidget = pyqtSignal(str, bool)
    searchColor = pyqtSignal(list, list)
    avatarColor = pyqtSignal(list)

    def __init__(self, title):
        super(FeatureWindow, self).__init__()
        self.title = title
        self.width = 640
        self.height = 480

        self.pipeline = ImagePipeline()
        self.running = False

        self.createResultWidget.connect(self.onCreateResultWidget)
        self.createEmptyWidget.connect(self.onCreateEmptyWidget)

        self.searchLayout = FlowLayout()
        self.searchList = []

        self.searchDict = {}

        self.videoPath = ""

        self.resultImageList = []
        self.resultDict = []

        self.initUI()

    def initUI(self):
        self.setGeometry(0,0, self.width, self.height)
        self.setWindowTitle(self.title)

        # Initialize tab screen
        self.tabs = QTabWidget(self)
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()

        # Add tabs
        self.tabs.addTab(self.tab1,"Preset 1")
        self.tabs.addTab(self.tab2,"Preset 2")
        self.tabs.addTab(self.tab3,"Preset 3")

        #frameInputLabel = QLabel("Grab a frame each x seconds: ")
        #self.frameInputEdit = QLineEdit()
        #self.frameInputEdit.returnPressed.connect(self.startAnalyse)

        changeAvatarButton = QPushButton("Change Avatar")
        changeAvatarButton.clicked.connect(self.onChangeAvatarClicked)

        changeAvatarButton.setStyleSheet(
            "padding-top: 10px; padding-bottom: 20px;"
            "margin-top: 5px; margin-bottom: 30px; margin-left: 50px; margin-right: 50px;")

        #hBox = QHBoxLayout()

        #hBox.addWidget(frameInputLabel)
        #hBox.addWidget(self.frameInputEdit)

        #searchText = QLabel("Search: ")
        #searchEdit = QLineEdit()
        #searchEdit.textChanged.connect()

        #hBox2 = QHBoxLayout()
        #hBox2.addWidget(searchText)
        #hBox2.addWidget(searchEdit)

        self.colorWheel = ColorCircleDialog()
        self.colorWheel.currentColorChanged.connect(self.onColorModified)

        self.avatarWidget = GLWidget()
        self.avatarColor.connect(self.avatarWidget.changeColor)
        self.avatarWidget.selectedAvatar.connect(self.avatarChanged)

        #hBox3 = QHBoxLayout()
        #vBox2 = QVBoxLayout()

        #topButton = QPushButton("1st Color")
        #bottomButton = QPushButton("2nd Color")
        #self.clearButton = QPushButton("Clear")
        #self.resetButton = QPushButton("Reset")
        
        #topButton.clicked.connect(self.avatarWidget.topSelected)
        #bottomButton.clicked.connect(self.avatarWidget.bottomSelected)
        #self.clearButton.clicked.connect(self.clearColor)

        #vBox2.addWidget(topButton)
        #vBox2.addWidget(bottomButton)
        #vBox2.addWidget(self.clearButton)
        #vBox2.addWidget(self.resetButton)
        #vBox2.addStretch()
        #hBox3.addLayout(vBox2)
        #hBox3.addWidget(self.colorWheel)

        vBox = QVBoxLayout()

        #vBox.addLayout(hBox)
        #vBox.addWidget(runButton)
        #vBox.addLayout(hBox2)
        #vBox.addLayout(self.searchLayout)
        #vBox.addLayout(hBox3)
        vBox.addWidget(self.avatarWidget)
        #vBox.addWidget(self.colorWheel)
        vBox.addWidget(changeAvatarButton)
        #vBox.addStretch()

        self.resultHolder = ResultHolder()
        self.searchColor.connect(self.resultHolder.colorSearch)
        self.resultHolder.sendVisibleList.connect(self.onReceiveVisibleList)

        #self.mainWidget = QWidget()
        self.setCentralWidget(self.tabs)

        self.layout = QHBoxLayout()

        self.layout.addLayout(vBox)
        self.layout.addWidget(self.resultHolder)

        self.tab1.setLayout(self.layout)

        #self.mainWidget.setLayout(self.layout)

    def onChangeAvatarClicked(self):
        colorAvatarDialog = ColorAvatarDialog(self)

        if colorAvatarDialog.exec():
            print("Success!")

            top, bottom, color, color2 = colorAvatarDialog.getClothes()
            self.avatarWidget.setSelectedClothes(top, bottom, color, color2)
        else:
            print("Cancel!")

    def showWidget(self):
        self.show()

    def setVideo(self, videoPath):
        self.stopAnalyse()

        self.videoPath = videoPath

        self.resultHolder.clear()

        if os.path.exists(os.path.splitext(self.videoPath)[0] + '.npz') and os.path.exists(os.path.splitext(self.videoPath)[0] + '.metadata'):
            images = np.load(os.path.splitext(self.videoPath)[0] + '.npz')

            with open(os.path.splitext(self.videoPath)[0] + ".metadata", 'rb') as handle:
                metadatas = pickle.load(handle)

            resultWidget = None

            lastFrameNumber = 0

            for x in range(len(metadatas)):
                metadata = metadatas[x]
                frame = images[images.files[(2 * x)]]
                fullFrame = images[images.files[(2 * x) + 1]]

                if lastFrameNumber != metadata['frame number'] and resultWidget is not None:
                    print(lastFrameNumber, metadata['frame number'])
                    self.resultHolder.appendWidget(resultWidget)
                    self.resultHolder.appendItem(QSpacerItem(100, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
                    lastFrameNumber = metadata['frame number']
                    resultWidget = None

                if resultWidget is None:
                    resultWidget = ResultWidget(metadata['frame time'], metadata['frame number'])

                label = OutfitLabel(metadata['text'], metadata['frame time'], metadata['frame number'], fullFrame, metadata['color'], metadata['box'], frame)
                label.sendInfo.connect(self.onOutfitLabelClicked)
                resultWidget.addWidget(label, metadata['text'])

            if resultWidget is not None:
                self.resultHolder.appendWidget(resultWidget)
                self.resultHolder.appendItem(QSpacerItem(100, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
                
        else:
            self.startAnalyse()

    @pyqtSlot()
    def startAnalyse(self):
        if self.running == False:
            self.threadAnalyse = threading.Thread(target = self.analyse) # Lance le thread cycleThread
            self.threadAnalyse.start()

    def stopAnalyse(self):
        self.running = False

    def runPipeline(self, images, count, fps):
        timeElapsed = 0.0

        t1 = time.perf_counter()
        #print('successfully grabbed frame')
        #frameCounted += 1

        t5 = time.perf_counter()
        #result = asyncio.run(self.pipeline.forward(images))
        result = self.pipeline.forward(images)
        t6 = time.perf_counter()

        print(f"Image forwarding took: {t6 - t5:0.4f} seconds for {len(images)} images")

        resultWidget = None
        if(len(result[0]) != 0 or len(result[1]) != 0):
            resultWidget = [(count, count/fps)]

        #Top Result
        for x in range(len(result[0])):
            #topBodyPrediction = [result[0][x]['prediction'][0][0], 0, result[0][x]['prediction'][0][2], result[0][x]['prediction'][0][3], result[0][x]['prediction'][0][4], 0, result[0][x]['prediction'][0][6]]

            print("top")
            #resultWidget.append((class_names[np.argmax(topBodyPrediction)], result[0][x]['image'], result[0][x]['color'], result[0][x]['box']))
            resultWidget.append(('Tshirt', result[0][x]['image'], result[0][x]['color'], result[0][x]['box'], result[0][x]['gray']))

        #Bottom Result
        for x in range(len(result[1])):
            print("bottom")

            resultWidget.append(('Trouser', result[1][x]['image'], result[1][x]['color'], result[1][x]['box'], result[1][x]['gray']))

        t3 = time.perf_counter()

        if resultWidget != None:
            self.createResultWidget.emit(resultWidget)
        else:
            self.createEmptyWidget.emit()

        t4 = time.perf_counter()

        print(f"Object Creation took: {t4 - t3:0.4f} seconds")

        t2 = time.perf_counter()
        timeElapsed += t2 - t1

        print(f"Forwarding took: {t2 - t1:0.4f} seconds")

        return timeElapsed

    def analyse(self):
        print("Running the pipeline on a Video")
        #fpsReading = int(self.frameInputEdit.text())
        fpsReading = int(1)

        self.running = True

        vidcap = cv2.VideoCapture(self.videoPath)
        count = 0
        success = True
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        frameCounted = 0

        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        t1 = time.perf_counter()

        timeElapsed = 0.0

        while success:
            success,image = vidcap.read()
            #print('read a new frame:',success)
            if count%(fps*int(fpsReading)) == 0 :
                frameCounted += 1

                timeElapsed += self.runPipeline(image, count, fps)

            count += 1
        
        t2 = time.perf_counter()
        print(f"Frame read: {(t2 - t1 - timeElapsed) / frameCounted:0.4f} seconds")

        print(f"Elapsed Time: {timeElapsed:0.2f}, Mean Time per image : {timeElapsed / frameCounted:0.4f}")

        self.running = False

        resultImages, resultMetadatas = self.resultHolder.getResultLists()

        print(resultMetadatas)

        np.savez_compressed(os.path.splitext(self.videoPath)[0], *(resultImages))

        with open(os.path.splitext(self.videoPath)[0] + ".metadata", 'wb') as handle:
            pickle.dump(resultMetadatas, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
                '''
                label = OutfitLabel(widgetInfo[x][0], widgetInit[0], widgetInfo[x][1], widgetInfo[x][2], widgetInfo[x][3])
                label.sendInfo.connect(self.onOutfitLabelClicked)
                #self.resetButton.clicked.connect(label.resetWidget)
                resultWidget.addWidget(label, widgetInfo[x][1], widgetInfo[x][0])
                '''

                label = OutfitLabel(widgetInfo[x][0], widgetInit[0], widgetInit[1], widgetInfo[x][1], widgetInfo[x][2], widgetInfo[x][3], widgetInfo[x][4])
                label.sendInfo.connect(self.onOutfitLabelClicked)
                #self.resetButton.clicked.connect(label.resetWidget)
                resultWidget.addWidget(label, widgetInfo[x][0])

                #self.resultHolder.appendWidget(label)

        #TODO uncomment this
        self.resultHolder.appendWidget(resultWidget)

        self.resultHolder.appendItem(QSpacerItem(100, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        resultWidget.showResult.connect(self.onWidgetClicked)
        resultWidget.sendVisibleList.connect(self.resultHolder.onWidgetVisibilityChanged)
        self.searchWidget.connect(resultWidget.onSearchWidget)
    
    @pyqtSlot()
    def onCreateEmptyWidget(self):
        self.resultHolder.appendEmpty()

    @pyqtSlot(QColor)
    def onColorModified(self, color):
        #self.searchColor.emit([color.hue(), color.saturation(), color.value()])
        #self.searchColor.emit([color.red(), color.green(), color.blue()])
        self.avatarColor.emit([color.red() / 255, color.green() / 255, color.blue() / 255])

    @pyqtSlot(int)
    def onWidgetClicked(self, frameNumber):
        self.setFrame.emit(frameNumber)

    @pyqtSlot(list)
    def onReceiveVisibleList(self, visibleList):
        self.sendVisibleList.emit(visibleList)

    @pyqtSlot(str, int, np.ndarray, np.ndarray)
    def onOutfitLabelClicked(self, string, frameNumber, frame, box):
        self.printFrame.emit(frameNumber, frame, box)

    @pyqtSlot(str)
    def onSearchLabelClicked(self, itemSearch):
        if self.searchDict[itemSearch] == False:
            self.searchDict[itemSearch] = True

            self.searchWidget.emit(itemSearch, True)
        else:
            self.searchDict[itemSearch] = False

            self.searchWidget.emit(itemSearch, False)

    @pyqtSlot(list, list, str)
    def avatarChanged(self, color1, color2, clothType):
        self.searchColor.emit([color1[0], color1[1], color1[2]], [color2[0], color2[1], color2[2]])

    def setCustomWidget(self, widget):
        self.setCentralWidget(widget)

    def resize(self):
        self.width = self.size().width()
        self.height = self.size().height()

    @pyqtSlot()
    def clearColor(self):
        self.avatarWidget.clearColor()
        #self.resetButton.clicked.emit()