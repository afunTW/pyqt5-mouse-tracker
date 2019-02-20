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
from PyQt5.QtGui import QPainter, QPixmap, QImage, QPen, QPalette, QPolygon

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
        self._w, self._h, self._c = (1280, 720, 3)
        self._pen_measurement = QPen(Qt.black, 4, cap=Qt.RoundCap)
        self._pen_measurement_line = QPen(Qt.black, 1)
        self._points = QPolygon()
        
        # init
        self.init_ui()
    
    def _ndarray_to_qimage(self, arr):
        w, h = arr.shape[:2]
        return QImage(arr.data, w, h, w*3, QImage.Format_RGB888)
    
    def _qimage_to_qpixmap(self, qimg):
        return QPixmap.fromImage(qimg)

    def _reset_qpixmap(self):
        self._pixmap = self._qimage_to_qpixmap(self._image)
    
    def init_ui(self):
        # set windows
        self.setWindowTitle(self.title)
        self.resize(self._w, self._h)
        frame_geo = self.frameGeometry()
        frame_geo.moveCenter(self.screen.center())
        self.move(frame_geo.topLeft())

        # set layout
        self.root = QGridLayout()
        self._hbox_body = QHBoxLayout()
        self.setLayout(self.root)
        self.show()

        # set widget
        if self._image:
            self._w, self.h = self._image.shape[:2]
        else:
            self._image = np.zeros((self._w, self._h, self._c), dtype='float32')
        self._image = self._ndarray_to_qimage(self._image)
        self._pixmap = self._qimage_to_qpixmap(self._image)
    
    def paintEvent(self, e):
        if self._points:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            for i in range(self._points.count()):
                painter.setPen(self._pen_measurement)
                painter.drawPoint(self._points.point(i))
                if i:
                    painter.setPen(self._pen_measurement_line)
                    painter.drawLine(self._points.point(i-1), self._points.point(i))
    
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drawing = True
            self._points = QPolygon()
    
    def mouseMoveEvent(self, e):
        if e.buttons() and Qt.LeftButton and self._drawing:
            self._points << e.pos()
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
