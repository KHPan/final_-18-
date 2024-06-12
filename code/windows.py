import HDR
import Stich
import cv2
import sys
from PyQt5.QtWidgets import (QWidget, QMainWindow, QApplication,
	QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
	QLabel, QPushButton, QComboBox, QLineEdit, QFileDialog)
from PyQt5.QtGui import QFont, QPixmap

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

	def addFile(self):
		filenames, _ = QFileDialog.getOpenFileNames(self, filter="圖片 (*.png *.bmp *.jpg)")
		for filename in filenames:
			label = QLabel(self)
			label.setPixmap(QPixmap(filename))
			self.picLayout.addWidget(label)

	def initUI(self):
		self.resize(1200, 800)
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
		buttonLayout.addStretch()
		mainLayout.addLayout(buttonLayout)

		mainLayout.addWidget(getLine())

		self.picLayout = QHBoxLayout()
		mainLayout.addStretch()
		mainLayout.addLayout(self.picLayout)

if __name__ == "__main__":
	app = QApplication(sys.argv)
	mainWindow = MainWindow()
	mainWindow.show()
	sys.exit(app.exec_())