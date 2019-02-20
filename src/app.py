import logging
import random
import numpy as np

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QDesktopWidget
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPainter, QPixmap, QImage, QPen, QPalette, QPolygon

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

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
        self._pen_mouse = QPen(Qt.black, 4, cap=Qt.RoundCap, style=Qt.DotLine)
        self._pen_mouse_line = QPen(Qt.black, 1)
        self._pen_measure = QPen(Qt.red, 4, cap=Qt.RoundCap, style=Qt.SolidLine)
        self._pen_measure_line = QPen(Qt.red, 1)
        self._pen_predict = QPen(Qt.blue, 4, cap=Qt.RoundCap, style=Qt.DotLine)
        self._pen_predict_line = QPen(Qt.blue, 1)
        self._mouse_points = QPolygon()
        self._measure_points = QPolygon()
        self._predict_points = QPolygon()
        
        # init
        self.init_ui()
    
    def _ndarray_to_qimage(self, arr):
        w, h = arr.shape[:2]
        return QImage(arr.data, w, h, w*3, QImage.Format_RGB888)
    
    def _qimage_to_qpixmap(self, qimg):
        return QPixmap.fromImage(qimg)

    def _reset_qpixmap(self):
        self._pixmap = self._qimage_to_qpixmap(self._image)
    
    def _reset_polygon(self):
        self._mouse_points = QPolygon()
        self._measure_points = QPolygon()
        self._predict_points = QPolygon()

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
    
    def init_kalman_filter(self, x: float, y: float):
        dim_x = 4   # Number of state variables (x, y, x_velocity, y_velocity)
        dim_z = 2   # Number of of measurement inputs (x, y)
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.kf.x = np.array([x, y, 0., 0.])        # init state (position, velocity) (dim_x, 1)
        self.kf.F = np.array([[1., 0., 1., 0.],     # state transition matrix (dim_x, dim_x)
                              [0., 1., 0., 1.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]])
        self.kf.H = np.array([[1., 0., 0., 0.],     # measurement function, ((dim_z, dim_x))
                              [0., 1., 0., 0.]])
        self.kf.P = np.eye(dim_x) * 1000.           # covariance matrix (dim_x, dim_x)
        self.kf.R = np.eye(dim_z) * 10.             # measurement noise covariance (dim_z, dim_z)
        self.kf.Q = Q_discrete_white_noise(         # process uncertainty  (dim_x, dim_x)
            dim=dim_x, dt=.1, var=.1)

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        assert self._mouse_points.count() == self._predict_points.count()
        for i in range(self._mouse_points.count()):
            painter.setPen(self._pen_measure)
            painter.drawPoint(self._mouse_points.point(i))
            painter.setPen(self._pen_predict)
            painter.drawPoint(self._predict_points.point(i))
            if i:
                painter.setPen(self._pen_measure_line)
                painter.drawLine(self._mouse_points.point(i-1), self._mouse_points.point(i))
                painter.setPen(self._pen_predict_line)
                painter.drawLine(self._predict_points.point(i-1), self._predict_points.point(i))
    
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drawing = True
            self.logger.info(f"========== Reset QPolygon ==========")
            x, y = float(e.x()), float(e.y())
            self._reset_polygon()
            self.init_kalman_filter(x, y)
    
    def mouseMoveEvent(self, e):
        if e.buttons() and Qt.LeftButton and self._drawing:
            self.logger.debug(f"Record {e.pos()}")

            # measurement
            self._mouse_points << e.pos()

            # predict
            self.kf.predict()
            predict_x, predict_y = self.kf.x[:2]
            predict_x, predict_y = int(predict_x), int(predict_y)
            self._predict_points << QPoint(predict_x, predict_y)
            self.logger.info(f"KF.x = {self.kf.x[:2]}; mouse = {e.x()}, {e.y()}")
            self.kf.update(np.array([e.x(), e.y()], dtype='float32').reshape((2, 1)))

            # draw
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
