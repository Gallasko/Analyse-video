import sys
import numpy as np
import OpenGL.GL as gl

import array
import math

from PyQt5 import QtGui
from PyQt5.QtCore import QSize, QPoint, QRect, Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QOpenGLWidget, QHBoxLayout, QGridLayout, QLabel

#from framework import *

vertexDim = 3
nVertices = 3

cubemap = np.array([-1.0,  1.0, -1.0,
                    -1.0, -1.0, -1.0,
                    1.0, -1.0, -1.0,
                    1.0, -1.0, -1.0,
                    1.0,  1.0, -1.0,
                    -1.0,  1.0, -1.0,

                    -1.0, -1.0,  1.0,
                    -1.0, -1.0, -1.0,
                    -1.0,  1.0, -1.0,
                    -1.0,  1.0, -1.0,
                    -1.0,  1.0,  1.0,
                    -1.0, -1.0,  1.0,

                    1.0, -1.0, -1.0,
                    1.0, -1.0,  1.0,
                    1.0,  1.0,  1.0,
                    1.0,  1.0,  1.0,
                    1.0,  1.0, -1.0,
                    1.0, -1.0, -1.0,

                    -1.0, -1.0,  1.0,
                    -1.0,  1.0,  1.0,
                    1.0,  1.0,  1.0,
                    1.0,  1.0,  1.0,
                    1.0, -1.0,  1.0,
                    -1.0, -1.0,  1.0,

                    -1.0,  1.0, -1.0,
                    1.0,  1.0, -1.0,
                    1.0,  1.0,  1.0,
                    1.0,  1.0,  1.0,
                    -1.0,  1.0,  1.0,
                    -1.0,  1.0, -1.0,

                    -1.0, -1.0, -1.0,
                    -1.0, -1.0,  1.0,
                    1.0, -1.0, -1.0,
                    1.0, -1.0, -1.0,
                    -1.0, -1.0,  1.0,
                    1.0, -1.0,  1.0], dtype='float32')

class App(QWidget):

    def __init__(self):
        super(App, self).__init__()

        self.glWidget = GLWidget()

        label = QLabel("T - Shirt")
        label.setAttribute(Qt.WA_TransparentForMouseEvents)

        mainLayout = QGridLayout()
        mainLayout.addWidget(self.glWidget, 0, 0)
        mainLayout.addWidget(label, 0, 0)
        self.setLayout(mainLayout)

        self.title = 'OpenGL Window - PyQt5'
        self.left = 20
        self.top = 30
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)

        #self.setGeometry(self.left, self.top, self.width, self.height)

class GLWidget(QOpenGLWidget):
    clicked = pyqtSignal()
    selectedAvatar = pyqtSignal(list, str)

    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)

        self.profile = QtGui.QOpenGLVersionProfile()
        self.profile.setVersion( 4, 3)

        self.xRot = 0
        self.yRot = 0
        self.zRot = 0

        self.zPos = 0
        # self.program = None

        self.lastPos = QPoint()

        self.timer = QTimer(self)
        #self.timer.timeout.connect(self.onTimeout)

        self.increasing = True
        self.topColorsVector = QtGui.QVector4D(0.0, 0.0, 0.0, 1.0)
        self.bottomColorsVector = QtGui.QVector4D(0.0, 0.0, 0.0, 1.0)

        self.colors = []

        for i in range(10):
            self.colors.append(QtGui.QVector4D(0.0, 0.0, 0.0, 1.0))

        self.colors2 = []

        for i in range(10):
            self.colors2.append(QtGui.QVector4D(0.0, 0.0, 0.0, 1.0))

        self.selected = "1 Color"
        self.selectedCloth = 0
        self.selectedTop = 0
        self.selectedBottom = 6

    @pyqtSlot()
    def onTimeout(self):
        if self.increasing:
            if self.colorsVector.x() < 1.0:
                self.colorsVector.setX(self.colorsVector.x() + 0.1)

            elif self.colorsVector.y() < 1.0:
                self.colorsVector.setY(self.colorsVector.y() + 0.1)

            elif self.colorsVector.z() < 1.0:
                self.colorsVector.setZ(self.colorsVector.z() + 0.1)

            else:
                self.increasing = False
        else:
            if self.colorsVector.x() > 0.0:
                self.colorsVector.setX(self.colorsVector.x() - 0.1)

            elif self.colorsVector.y() > 0.0:
                self.colorsVector.setY(self.colorsVector.y() - 0.1)

            elif self.colorsVector.z() > 0.0:
                self.colorsVector.setZ(self.colorsVector.z() - 0.1)

            else:
                self.increasing = True

        self.update()
        self.paintGL()

    @pyqtSlot(list)
    def changeColor(self, colors):
        if(self.selected == "1 Color"):
            self.colors[self.selectedCloth].setX(colors[0])
            self.colors[self.selectedCloth].setY(colors[1])
            self.colors[self.selectedCloth].setZ(colors[2])
        if(self.selected == "2 Color"):
            self.colors2[self.selectedCloth].setX(colors[0])
            self.colors2[self.selectedCloth].setY(colors[1])
            self.colors2[self.selectedCloth].setZ(colors[2])

        self.update()
        self.paintGL()

    @pyqtSlot()
    def topSelected(self):
        self.selected = "1 Color"

    @pyqtSlot()
    def bottomSelected(self):
        self.selected = "2 Color"
    
    def getOpenglInfo(self):
        info = """
            Vendor: {0}
            Renderer: {1}
            OpenGL Version: {2}
            Shader Version: {3}
        """.format(
            gl.glGetString(gl.GL_VENDOR),
            gl.glGetString(gl.GL_RENDERER),
            gl.glGetString(gl.GL_VERSION),
            gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)
        )

        return info


    def rotateBy(self, xAngle, yAngle, zAngle):
        print(str(xAngle) + ", " + str(yAngle) + ", " + str(zAngle))
        self.xRot += xAngle
        self.yRot += yAngle
        self.zRot += zAngle
        self.update()
    '''
    def renderText(self, x, y, z, str, font = QtGui.QFont()):
        textPosX = x
        textPosY = y
        textPosZ = 0.0

        glColor = [0.0, 0.0, 0.0, 0.0]
        gl.glGetDoublev(gl.GL_CURRENT_COLOR, glColor)
        fontColor = QtGui.QColor(glColor[0], glColor[1], glColor[2], glColor[3])

        painter = QtGui.QPainter(self)
        painter.setPen(fontColor)
        painter.setFont(font)
        painter.drawText(textPosX, textPosY, str)
        painter.end()
    '''
    '''
        #// Identify x and y locations to render text within widget
        int height = this->height()
        GLdouble textPosX = 0, textPosY = 0, textPosZ = 0
        project(x, y, 0f, &textPosX, &textPosY, &textPosZ)
        textPosY = height - textPosY; // y is inverted

        // Retrieve last OpenGL color to use as a font color
        GLdouble glColor[4];
        glGetDoublev(GL_CURRENT_COLOR, glColor);
        QColor fontColor = QColor(glColor[0], glColor[1], glColor[2], glColor[3]);

        // Render text
        QPainter painter(this);
        painter.setPen(fontColor);
        painter.setFont(font);
        painter.drawText(textPosX, textPosY, text);
        painter.end()
    '''


    def minimumSizeHint(self):
        return QSize(600, 400)

    def initializeGL(self):
        print(self.getOpenglInfo())

        self.gl = self.context().versionFunctions( self.profile )

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # Vertex Array Object
        self.vao = QtGui.QOpenGLVertexArrayObject( self )
        self.vao.create()

        # Set up and link shaders
        self.vsShader = QtGui.QOpenGLShader(QtGui.QOpenGLShader.Vertex, self)
        self.vsShader.compileSourceFile('Shader/shader.vs')

        self.fsShader = QtGui.QOpenGLShader(QtGui.QOpenGLShader.Fragment, self)
        self.fsShader.compileSourceFile('Shader/shader.fs')

        self.program = QtGui.QOpenGLShaderProgram( self )
        self.program.addShader(self.vsShader)
        self.program.addShader(self.fsShader)
        self.program.link()

        self.vao.bind()


        self.vertices = np.array([ -0.5,  0.5, 0.0,# x, y, z
                                    0.5,  0.5, 0.0,
                                   -0.5, -0.5, 0.0,
                                    0.5, -0.5, 0.0], dtype='float32')

        self.texCoord = np.array([ 0.0, 1.0,# x, y, z
                                   1.0, 1.0,
                                   0.0, 0.0,
                                   1.0, 0.0 ], dtype='float32')

        #self.colors = np.array([ 0.0, 1.0, 0.0, 0.8,
        #                         0.0, 1.0, 0.0, 0.8,
        #                         0.0, 1.0, 0.0, 0.8,
        #                         0.0, 1.0, 0.0, 0.8 ], dtype='float32')

        self.indices = np.array([0, 1, 2, 1, 2, 3], dtype='uint32')

        
        '''
        vbo = QtGui.QOpenGLBuffer( QtGui.QOpenGLBuffer.VertexBuffer )
        vbo.create()
        vbo.bind()

        vertices = np.array( self.vertices, np.float32 )
        vbo.allocate( vertices, vertices.shape[0] * vertices.itemsize )

        attr_loc = self.program.attributeLocation( "aPos" )
        self.program.enableAttributeArray( attr_loc )
        self.program.setAttributeBuffer( attr_loc, gl.GL_FLOAT, 0, 3 )
        
        attr_loc = self.program.attributeLocation( "aTexCoord" )
        self.program.enableAttributeArray( attr_loc )
        self.program.setAttributeBuffer( attr_loc, gl.GL_FLOAT, 0, 2 )
        #self.program.enableVertexAttributeArray(0)
        #self.program.
        vbo.release()
        '''

        self.vboVertices = self.setVertexBuffer( self.vertices, 3, self.program, "aPos" )

        self.vboTex = self.setVertexBuffer( self.texCoord, 2, self.program, "aTexCoord" )

        #self.vboColors = self.setVertexBuffer (self.colors, 4, self.program, "aColors") 

        self.ebo = QtGui.QOpenGLBuffer( QtGui.QOpenGLBuffer.IndexBuffer )
        self.ebo.create()
        self.ebo.bind()

        self.ebo.allocate(self.indices, self.indices.shape[0] * self.indices.itemsize )

        self.vao.release()

        self.program.bind()

        self.program.release()

        self.textures = []

        img = QtGui.QImage("Body/Hoodie.png")

        self.textures.append(QtGui.QOpenGLTexture(img))
        self.textures[0].setMinMagFilters(QtGui.QOpenGLTexture.Linear, QtGui.QOpenGLTexture.Linear)
        self.textures[0].setWrapMode(QtGui.QOpenGLTexture.ClampToEdge)

        img = QtGui.QImage("Body/Jacket.png")

        self.textures.append(QtGui.QOpenGLTexture(img))
        self.textures[1].setMinMagFilters(QtGui.QOpenGLTexture.Linear, QtGui.QOpenGLTexture.Linear)
        self.textures[1].setWrapMode(QtGui.QOpenGLTexture.ClampToEdge)

        img = QtGui.QImage("Body/Lab Coat.png")

        self.textures.append(QtGui.QOpenGLTexture(img))
        self.textures[2].setMinMagFilters(QtGui.QOpenGLTexture.Linear, QtGui.QOpenGLTexture.Linear)
        self.textures[2].setWrapMode(QtGui.QOpenGLTexture.ClampToEdge)

        img = QtGui.QImage("Body/Long Sleeve.png")

        self.textures.append(QtGui.QOpenGLTexture(img))
        self.textures[3].setMinMagFilters(QtGui.QOpenGLTexture.Linear, QtGui.QOpenGLTexture.Linear)
        self.textures[3].setWrapMode(QtGui.QOpenGLTexture.ClampToEdge)

        img = QtGui.QImage("Body/Turtle Neck.png")

        self.textures.append(QtGui.QOpenGLTexture(img))
        self.textures[4].setMinMagFilters(QtGui.QOpenGLTexture.Linear, QtGui.QOpenGLTexture.Linear)
        self.textures[4].setWrapMode(QtGui.QOpenGLTexture.ClampToEdge)

        img = QtGui.QImage("Body/Long Sleeve.png")

        self.textures.append(QtGui.QOpenGLTexture(img))
        self.textures[5].setMinMagFilters(QtGui.QOpenGLTexture.Linear, QtGui.QOpenGLTexture.Linear)
        self.textures[5].setWrapMode(QtGui.QOpenGLTexture.ClampToEdge)

        img = QtGui.QImage("Bottom/Standing/Baggy Pants.png")

        self.textures.append(QtGui.QOpenGLTexture(img))
        self.textures[6].setMinMagFilters(QtGui.QOpenGLTexture.Linear, QtGui.QOpenGLTexture.Linear)
        self.textures[6].setWrapMode(QtGui.QOpenGLTexture.ClampToEdge)

        img = QtGui.QImage("Bottom/Standing/Shorts.png")

        self.textures.append(QtGui.QOpenGLTexture(img))
        self.textures[7].setMinMagFilters(QtGui.QOpenGLTexture.Linear, QtGui.QOpenGLTexture.Linear)
        self.textures[7].setWrapMode(QtGui.QOpenGLTexture.ClampToEdge)

        img = QtGui.QImage("Bottom/Standing/Skinny Jeans.png")

        self.textures.append(QtGui.QOpenGLTexture(img))
        self.textures[8].setMinMagFilters(QtGui.QOpenGLTexture.Linear, QtGui.QOpenGLTexture.Linear)
        self.textures[8].setWrapMode(QtGui.QOpenGLTexture.ClampToEdge)

        img = QtGui.QImage("Bottom/Standing/Skirt.png")

        self.textures.append(QtGui.QOpenGLTexture(img))
        self.textures[9].setMinMagFilters(QtGui.QOpenGLTexture.Linear, QtGui.QOpenGLTexture.Linear)
        self.textures[9].setWrapMode(QtGui.QOpenGLTexture.ClampToEdge)

        self.bottomTexture = self.textures[9]
        self.topTexture = self.textures[0]

        self.timer.start(100)


    def resizeGL(self, width, height):
        gl.glViewport(0, 0, width, height)

    def setVertexBuffer( self, data_array, dim_vertex, program, shader_str ):
        vbo = QtGui.QOpenGLBuffer( QtGui.QOpenGLBuffer.VertexBuffer )
        vbo.create()
        vbo.bind()

        vertices = np.array( data_array, np.float32 )
        vbo.allocate( vertices, vertices.shape[0] * vertices.itemsize )

        attr_loc = program.attributeLocation( shader_str )
        program.enableAttributeArray( attr_loc )
        program.setAttributeBuffer( attr_loc, gl.GL_FLOAT, 0, dim_vertex )
        vbo.release()

        return vbo


    def paintGL(self):
        #gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(84 / 255.0, 88 / 255.0, 160 / 255.0, 0.0)

        self.program.bind()

        #self.texture = QtGui.QOpenGLTexture(QtGui.QOpenGLTexture.TargetCubeMap)
        #self.texture.create()
        #print(self.texture.isCreated())

        #self.texture.setSize(img.width(), img.height())
        #self.texture.setFormat(QtGui.QOpenGLTexture.RGBAFormat)

        #self.texture.allocateStorage()

        '''
        self.texture.setData(0, 0, QtGui.QOpenGLTexture.CubeMapNegativeX, QtGui.QOpenGLTexture.RGBA, QtGui.QOpenGLTexture.UInt16, img.bits())
        self.texture.setData(0, 0, QtGui.QOpenGLTexture.CubeMapNegativeY, QtGui.QOpenGLTexture.RGBA, QtGui.QOpenGLTexture.UInt16, img.bits())
        self.texture.setData(0, 0, QtGui.QOpenGLTexture.CubeMapNegativeZ, QtGui.QOpenGLTexture.RGBA, QtGui.QOpenGLTexture.UInt16, img.bits())
        self.texture.setData(0, 0, QtGui.QOpenGLTexture.CubeMapPositiveX, QtGui.QOpenGLTexture.RGBA, QtGui.QOpenGLTexture.UInt16, img.bits())
        self.texture.setData(0, 0, QtGui.QOpenGLTexture.CubeMapPositiveY, QtGui.QOpenGLTexture.RGBA, QtGui.QOpenGLTexture.UInt16, img.bits())
        self.texture.setData(0, 0, QtGui.QOpenGLTexture.CubeMapPositiveZ, QtGui.QOpenGLTexture.RGBA, QtGui.QOpenGLTexture.UInt16, img.bits())
        '''

        self.vao.bind()

        # Avatar Tex

        # initialise Camera matrix with initial rotation
        m = QtGui.QMatrix4x4()
        
        m.ortho(-1.0, 1.0, 1.0, -1.0, -2.0, 15.0)
        m.scale(256.0 / self.width(), 187.0 / self.height(), 1.0)
        m.translate(-1.0, 0.5, self.zPos - 1)
        
        #m.scale(256.0 / self.width(), 187.0 / self.height(), 1.0)  
        m.rotate(self.xRot / 16.0, 1.0, 0.0, 0.0)
        m.rotate(self.yRot / 16.0, 0.0, 1.0, 0.0)
        m.rotate(self.zRot / 16.0, 0.0, 0.0, 1.0)

        self.bottomTexture.bind()
        
        self.program.setUniformValue('mvp', m)
        self.program.setUniformValue('texture', 0)
        self.program.setUniformValue('colors', self.colors[self.selectedBottom])

        gl.glDrawElements( gl.GL_TRIANGLES, 6*6, gl.GL_UNSIGNED_INT, None)

        m.setToIdentity()

        # initialise Camera matrix with initial rotation
        m = QtGui.QMatrix4x4()
        
        m.ortho(-1.0, 1.0, 1.0, -1.0, -2.0, 15.0)
        m.scale(256.0 / self.width(), 187.0 / self.height(), 1.0)
        m.translate(-1.0, 0.0, self.zPos)
        #m.scale(256.0 / self.width(), 187.0 / self.height(), 1.0)
        m.rotate(self.xRot / 16.0, 1.0, 0.0, 0.0)
        m.rotate(self.yRot / 16.0, 0.0, 1.0, 0.0)
        m.rotate(self.zRot / 16.0, 0.0, 0.0, 1.0)

        self.topTexture.bind()
        
        self.program.setUniformValue('mvp', m)
        self.program.setUniformValue('texture', 0)
        self.program.setUniformValue('colors', self.colors[self.selectedTop])

        gl.glDrawElements( gl.GL_TRIANGLES, 6*6, gl.GL_UNSIGNED_INT, None)

        #END Avatar Tex

        #Hud Tex

        for x in range(10):
            m.setToIdentity()

            # initialise Camera matrix with initial rotation
            m = QtGui.QMatrix4x4()
            
            m.ortho(-1.0, 1.0, 1.0, -1.0, -2.0, 15.0)
            m.scale(256.0 / self.width(), 187.0 / self.height(), 1.0)
            m.translate(1.0 + (x % 2), -2.0 + math.floor(x / 2), self.zPos)
            #m.scale(256.0 / self.width(), 187.0 / self.height(), 1.0)
            #m.rotate(self.xRot / 16.0, 1.0, 0.0, 0.0)
            #m.rotate(self.yRot / 16.0, 0.0, 1.0, 0.0)
            #m.rotate(self.zRot / 16.0, 0.0, 0.0, 1.0)

            self.textures[x].bind()
            
            self.program.setUniformValue('mvp', m)
            self.program.setUniformValue('texture', 0)
            self.program.setUniformValue('colors', self.colors[x])
            '''
            if(x < 6):
                self.program.setUniformValue('colors', self.topColorsVector)
            else:
                self.program.setUniformValue('colors', self.bottomColorsVector)
            '''

            gl.glDrawElements( gl.GL_TRIANGLES, 6*6, gl.GL_UNSIGNED_INT, None)


        self.vao.release()

        self.program.release()

    '''
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawText(100,100, "Hello PyQt5 App Development")
 
        rect = QRect(100,150, 250,25)
        painter.drawRect(rect)
        painter.drawText(rect, Qt.AlignCenter, "Hello World")

        #self.update()
        #self.paintGL()
    '''

    def mousePressEvent(self, event):
        self.lastPos = event.pos()
        print(event.pos())
        toScreen = [2.0 * event.pos().x() / self.width() - 1.0, 2.0 * event.pos().y() / self.height() - 1.0]
        print( toScreen[0], toScreen[1] )
        #toScreen[0] = toScreen[0] * 256.0 / self.width()
        #toScreen[1] = toScreen[1] * 187.0 / self.height()
        #print( toScreen[0], toScreen[1] )
        #print ( -1.0 * 256.0 / self.width() , 0.5 * 187.0 / self.height())

        for x in range(10):
            texCoord = [(1.0 + (x % 2) + 0.5) * 256.0 / self.width(), (1.0 + (x % 2) - 0.5) * 256.0 / self.width(), (-2.0 + math.floor(x / 2) + 0.5) * 187.0 / self.height(), (-2.0 + math.floor(x / 2) - 0.5) * 187.0 / self.height()]
            print(texCoord)

            if (toScreen[0] <= texCoord[0] and 
                toScreen[0] >= texCoord[1] and
                toScreen[1] <= texCoord[2] and
                toScreen[1] >= texCoord[3]):
                self.selectedCloth = x
                self.selectedAvatar.emit([self.colors[self.selectedCloth].x() * 255, self.colors[self.selectedCloth].y() * 255, self.colors[self.selectedCloth].z() * 255], "TODO l 548")
                if x <= 5:
                    self.selectedTop = x
                    self.topTexture = self.textures[x]
                else:
                    self.selectedBottom = x
                    self.bottomTexture = self.textures[x]

        self.update()
        self.paintGL()


    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()
        
        #if event.buttons() & Qt.LeftButton:
        #    self.rotateBy(8 * dy, 8 * dx, 0)
        #elif event.buttons() & Qt.RightButton:
        #    self.rotateBy(8 * dy, 0, 8 * dx)

        self.lastPos = event.pos()
        self.paintGL()
        

    def wheelEvent(self, event):
        self.zPos += event.angleDelta().y() / 120
        print(self.zPos)

        self.update()
        self.paintGL()

    def mouseReleaseEvent(self, event):
        self.clicked.emit()

if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())