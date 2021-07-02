import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5 import QtGui
from PyQt5.QtOpenGL import *
from PyQt5 import QtCore, QtWidgets, QtOpenGL

import ctypes
import numpy as np
import OpenGL.arrays.vbo as glvbo

import OpenGL.GL as gl

vertex_code = '''
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

uniform mat4 model;
uniform mat4 scale;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	gl_Position = projection * view * model * scale * vec4(aPos, 1.0f);
	TexCoord = vec2(aTexCoord.x, aTexCoord.y);
}
'''

fragment_code = '''
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

// texture samplers
uniform sampler2D texture1;

void main()
{
	FragColor = texture(texture1, TexCoord);
}
'''

class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__()
        self.widget = glWidget()
        #self.button = QtWidgets.QPushButton('Test', self)
        mainLayout = QtWidgets.QHBoxLayout()
        mainLayout.addWidget(self.widget)
        #mainLayout.addWidget(self.button)
        self.setLayout(mainLayout)


class glWidget(QGLWidget):
    def __init__(self, parent=None):
        QGLWidget.__init__(self, parent)
        self.setMinimumSize(640, 480)

    def paintGL(self):
        #gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        gl.glLoadIdentity()
        gl.glRotated(50.0, 0.0, 1.0, 0.0)

        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)

    def initializeGL(self):
        gl.glClearColor(0., 0., 0.8, 0.)
        #glClearDepth(1.0)              
        #glDepthFunc(GL_LESS)
        #glEnable(GL_DEPTH_TEST)
        #glShadeModel(GL_SMOOTH)
        #glMatrixMode(GL_PROJECTION)
        #glLoadIdentity()                    
        #gluPerspective(45.0,1.33,0.1, 100.0) 
        #glMatrixMode(GL_MODELVIEW)

        program = gl.glCreateProgram()
        vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)

        # Set shaders source
        gl.glShaderSource(vertex, vertex_code)
        gl.glShaderSource(fragment, fragment_code)

        # Compile shaders
        gl.glCompileShader(vertex)
        if not gl.glGetShaderiv(vertex, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(vertex).decode()
            logger.error("Vertex shader compilation error: %s", error)

        gl.glCompileShader(fragment)
        if not gl.glGetShaderiv(fragment, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(fragment).decode()
            print(error)
            raise RuntimeError("Fragment shader compilation error")

        gl.glAttachShader(program, vertex)
        gl.glAttachShader(program, fragment)
        gl.glLinkProgram(program)

        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            print(gl.glGetProgramInfoLog(program))
            raise RuntimeError('Linking error')

        gl.glDetachShader(program, vertex)
        gl.glDetachShader(program, fragment)

        #gl.glUseProgram(program)


        self.vertices = np.array([
            # <- x,y,z ----->  <- r,g,b -->
            -1.0, -1.0, 0.0, 1.0, 0.0, 0.0,
            -1.0,  1.0, 0.0, 1.0, 0.0, 0.0,
             1.0, -1.0, 0.0, 1.0, 0.0, 0.0,
            -1.0,  1.0, 0.0, 0.0, 1.0, 0.0,
             1.0,  1.0, 0.0, 0.0, 1.0, 0.0,
             1.0, -1.0, 0.0, 0.0, 1.0, 0.0,
        ], 'f')

        self.vbo = glvbo.VBO(self.vertices)
        self.vbo.bind()

        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)

        buffer_offset = ctypes.c_void_p
        stride = (3+3)*self.vertices.itemsize
        gl.glVertexPointer(3, gl.GL_FLOAT, stride, None)
        gl.glColorPointer(3, gl.GL_FLOAT, stride, buffer_offset(12))

        gl.glBindVertexArray(0)


if __name__ == '__main__':    
    app = QtWidgets.QApplication(sys.argv)    
    Form = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(Form)    
    ui.show()    
    sys.exit(app.exec_())