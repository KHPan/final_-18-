import HDR
import Stich
import cv2
import sys
from PyQt5.QtWidgets import (QWidget, QMainWindow, QApplication,
	QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
	QLabel, QPushButton, QComboBox, QLineEdit, QFileDialog, QScrollArea)
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np

def getLine() -> QFrame:
	line = QFrame()
	line.setFrameShape(QFrame.HLine)
	line.setFrameShadow(QFrame.Sunken)
	return line

class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.imgs = []
		self.initUI()
	
	def loadPixmap(self, image: np.ndarray) -> QPixmap:
		height, width, channel = image.shape
		bytesPerLine = image.shape[2] * width
		qImg = QImage(image.data, width, height, bytesPerLine,
				QImage.Format_RGB888 if image.shape[2] == 3 else
					QImage.Format_RGBA8888).rgbSwapped()
		return QPixmap.fromImage(qImg).scaledToHeight(
					self.scrollArea.size().height() - 50,
					Qt.SmoothTransformation)

	def addFile(self):
		filenames, _ = QFileDialog.getOpenFileNames(self, filter="圖片 (*.png *.bmp *.jpg)")
		for filename in filenames:
			img = HDR.imread(filename)
			label = QLabel(self)
			label.setPixmap(self.loadPixmap(img))
			self.picLayout.addWidget(label)
			self.imgs.append(img)

	def initUI(self):
		self.resize(1200, 800)
		self.setSizePolicy
		self.setWindowTitle("HDR and stiching program")
		font = QFont()
		font.setPointSize(15)
		self.setFont(font)
		mainWidget = QWidget(self)
		self.setCentralWidget(mainWidget)
		mainLayout = QVBoxLayout(mainWidget)

		buttonLayout = QHBoxLayout()
		button = QPushButton("新增圖片", self)
		button.clicked.connect(self.addFile)
		buttonLayout.addWidget(button)
		buttonLayout.setAlignment(Qt.AlignLeft)
		mainLayout.addLayout(buttonLayout)

		mainLayout.addWidget(getLine())

		self.scrollArea = QScrollArea(self)
		imageContaner = QWidget()
		self.picLayout = QHBoxLayout(imageContaner)
		self.picLayout.setAlignment(Qt.AlignLeft)
		self.scrollArea.setWidget(imageContaner)
		self.scrollArea.setWidgetResizable(True)
		mainLayout.addWidget(self.scrollArea)

	def resizeEvent(self, event):
		img_index = 0
		for i in range(self.picLayout.count()):
			label = self.picLayout.itemAt(i).widget()
			if isinstance(label, QLabel):
				label.setPixmap(self.loadPixmap(self.imgs[img_index]))
				img_index += 1
		super().resizeEvent(event)

if __name__ == "__main__":
	app = QApplication(sys.argv)
	mainWindow = MainWindow()
	mainWindow.show()
	sys.exit(app.exec_())