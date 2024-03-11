import sys

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QGraphicsPixmapItem, QGraphicsItem, QMessageBox, QGraphicsScene
import cv2
from ultralytics import YOLO
from PIL import Image


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1035, 695)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(0, 0, 640, 640))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(660, 70, 341, 22))
        self.horizontalSlider.setAutoFillBackground(False)
        self.horizontalSlider.setMinimum(10)
        self.horizontalSlider.setMaximum(100)
        self.horizontalSlider.setProperty("value", 25)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.horizontalSlider.setTickInterval(10)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalSlider.valueChanged.connect(self.updateLabel)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(660, 50, 331, 16))
        self.label.setObjectName("label")
        self.openButton = QtWidgets.QPushButton(self.centralwidget)
        self.openButton.setGeometry(QtCore.QRect(660, 10, 361, 28))
        self.openButton.setObjectName("pushButton")
        self.saveButton = QtWidgets.QPushButton(self.centralwidget)
        self.saveButton.setGeometry(QtCore.QRect(660, 610, 361, 28))
        self.saveButton.setObjectName("pushButton_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1035, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.path = None
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def updateLabel(self, value):
        self.label.setText("Confidence " + str(value/100))

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Сегментация льдов"))
        self.label.setText(_translate("MainWindow", "Confidence " + str(self.horizontalSlider.value()/100)))
        self.openButton.setText(_translate("MainWindow", "Открыть изображение"))
        self.saveButton.setText(_translate("MainWindow", "Предсказать"))


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.openButton.clicked.connect(self.openButtonClicked)
        self.saveButton.clicked.connect(self.saveButtonClicked)

    def openButtonClicked(self):
        file_name = QFileDialog.getOpenFileName(self, 'Open file', '.', 'Image Files (*.png *.jpg *.bmp)')
        image_path = file_name[0]
        if (file_name[0] == ""):
            QMessageBox.information(self, "Prompt", self.tr("No picture file selected!"))
            return

        self.path = image_path
        print(self.path)
        img = cv2.imread(image_path)  # read image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image channel

        img = cv2.resize(img,(640,640))

        x = img.shape[1]  # Get image size
        y = img.shape[0]
        self.zoomscale = 1  # Image zoom scale

        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # Create pixel image element
        self.scene = QGraphicsScene()  # Create a scene
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.show()

    def saveButtonClicked(self):
        print(self.horizontalSlider.value()/100)
        print("Afsffafsa")
        print(self.path)
        if self.path is None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Выберите изображение")
            msg.setWindowTitle("Ошибка")
            msg.exec_()
            return

        model = YOLO('best (1).pt')  # load a custom model
        results = model.predict(source=self.path, imgsz=640, conf=self.horizontalSlider.value()/100, save=True
                                , show_conf=False, show_labels=False, line_width=0)  # predict on an image

        if results[0].masks is None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("C текущим уровнем уверенности лёд не обнаружен")
            msg.setWindowTitle("Ошибка")
            msg.exec_()
            return

        # Convert mask to single channel image
        mask_raw = results[0].masks[0].cpu().data.numpy().transpose(1, 2, 0)

        # Convert single channel grayscale to 3 channel image
        mask_3channel = cv2.merge((mask_raw, mask_raw, mask_raw))

        # Get the size of the original image (height, width, channels)
        h2, w2, c2 = results[0].orig_img.shape

        # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
        mask = cv2.resize(mask_3channel, (w2, h2))

        # Convert BGR to HSV
        hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

        # Define range of brightness in HSV
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([0, 0, 1])

        # Create a mask. Threshold the HSV image to get everything black
        mask = cv2.inRange(mask, lower_black, upper_black)

        # Invert the mask to get everything but black
        mask = cv2.bitwise_not(mask)

        # Apply the mask to the original image
        masked = cv2.bitwise_and(results[0].orig_img, results[0].orig_img, mask=mask)
        masked = cv2.resize(masked, (640, 640))
        # Show the masked part of the image
        cv2.imshow("mask", masked)


app = QtWidgets.QApplication([])
application = Window()
application.show()

sys.exit(app.exec())
