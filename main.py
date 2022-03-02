import sys

from PyQt5.QtWidgets import QApplication
from lib.playerwindow import PlayerWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    player = PlayerWindow()
    player.resize(640, 480)
    player.show()
    
    sys.exit(app.exec_())