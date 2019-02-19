import logging
import random
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QDesktopWidget
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPainter, QPixmap, QImage, QPen, QPalette

class KalmanFilterTracker(QWidget):
    def __init__(self, title='', image=''):
        super().__init__()
        self.title = title
        self.desktop = QDesktopWidget()
        self.screen = self.desktop.availableGeometry()
        self.logger = logging.getLogger(self.__class__.__name__)

        # drawing setting
        self._drawing = False
        self._image = image
        self._w, self._h = (1280, 720)
        self._pen_measurement = QPen(Qt.black, 3)
        self._pen_measurement_line = QPen(Qt.black, 1)
        
        # init
        self.init_ui()
    
    def _ndarray_to_qimage(self, arr):
        w, h = arr.shape[:2]
        return QImage(arr.data, w, h, w*3, QImage.Format_RGB888)
    
    def _qimage_to_qpixmap(self, qimg):
        return QPixmap.fromImage(qimg)
    
    def init_ui(self):
        # set windows
        self.setWindowTitle(self.title)
        self.resize(self._w, self._h)
        frame_geo = self.frameGeometry()
        frame_geo.moveCenter(self.screen.center())
        self.move(frame_geo.topLeft())
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(palette)


        # set layout
        self.root = QGridLayout()
        self._hbox_body = QHBoxLayout()
        self.setLayout(self.root)
        self.show()

        # set widget
        if self._image:
            self._w, self.h = self._image.shape
        else:
            self._image = np.zeros((self._w, self._h), dtype='float32')
        self._image = self._ndarray_to_qimage(self._image)
        self._pixmap = self._qimage_to_qpixmap(self._image)
    
    def paintEvent(self, e):
        # painter setting
        painter = QPainter(self)
        # painter.setRenderHint(QPainter.Antialiasing)
        # painter.drawPixmap(self.rect(), self._pixmap)
    
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drawing = True
            self._last_pts = e.pos()
    
    def mouseMoveEvent(self, e):
        if e.buttons() and Qt.LeftButton and self._drawing:
            self.logger.info(f"draw {e.pos()}")
            painter = QPainter(self)
            # painter.setPen(self._pen_measurement)
            # painter.drawPoint(e.pos())
            painter.setPen(self._pen_measurement_line)
            painter.drawLine(self._last_pts, e.pos())
            self._last_pts = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drawing = False
            

if __name__ == '__main__':
    import sys
    from utils import log_handler
    app = QApplication(sys.argv)
    viewer = KalmanFilterTracker()
    log_handler(viewer.logger)
    app.exec()
