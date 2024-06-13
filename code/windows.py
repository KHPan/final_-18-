import HDR
import Stich
import cv2
import sys
from PyQt5.QtWidgets import (QWidget, QMainWindow, QApplication,
	QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QScrollArea,
	QLabel, QPushButton, QComboBox, QLineEdit,
	QFileDialog, QMessageBox)
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np
from typing import Callable

def getLine() -> QFrame:
	line = QFrame()
	line.setFrameShape(QFrame.HLine)
	line.setFrameShadow(QFrame.Sunken)
	return line

class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.imgs = []
		self.result = None
		self.initUI()		
		self.sourceMode()
	
	def loadPixmap(self, image: np.ndarray) -> QPixmap:
		height, width, channel = image.shape
		bytesPerLine = image.shape[2] * width
		qImg = QImage(image.data, width, height, bytesPerLine,
				QImage.Format_RGB888 if image.shape[2] == 3 else
					QImage.Format_RGBA8888).rgbSwapped()
		return QPixmap.fromImage(qImg).scaledToHeight(
					self.scrollArea.size().height() - 50,
					Qt.SmoothTransformation)

	def addPic(self, img: np.ndarray):
		label = QLabel(self)
		label.setPixmap(self.loadPixmap(img))
		self.picLayout.addWidget(label)

	def addFile(self):
		filenames, _ = QFileDialog.getOpenFileNames(self,
										filter="圖片 (*.png *.bmp *.jpg)")
		for filename in filenames:
			img = HDR.imread(filename)
			self.addPic(img)
			self.imgs.append(img)

	def makeHDR(self):
		if len(self.imgs) <= 1:
			QMessageBox.warning(self, "警告", "兩個以上的圖片才能製造HDR影像")
			return
		self.result = HDR.makeHDR(self.imgs)
		self.resultMode()

	def stiching(self):
		if len(self.imgs) <= 1:
			QMessageBox.warning(self, "警告", "兩個以上的圖片才能組合全景圖")
			return
		self.result = Stich.stich(self.imgs)
		self.resultMode()
	
	def switchMode(self):
		if self.switchBtn.text() == "結果模式":
			self.resultMode()
		else:
			self.sourceMode()

	def saveResult(self):
		filename, _ = QFileDialog.getSaveFileName(self,
						filter="PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)")
		if filename:
			HDR.imwrite(filename, self.result)

	def resultMode(self):
		while self.picLayout.count():
			self.picLayout.takeAt(0).widget().deleteLater()
		self.addPic(self.result)
		self.switchBtn.setEnabled(True)
		self.switchBtn.setText("源圖模式")
		for btn in self.srcBtns:
			btn.setEnabled(False)
		for btn in self.resultBtns:
			btn.setEnabled(True)
	
	def sourceMode(self):
		while self.picLayout.count():
			self.picLayout.takeAt(0).widget().deleteLater()
		for img in self.imgs:
			self.addPic(img)
		self.switchBtn.setEnabled(self.result is not None)
		self.switchBtn.setText("結果模式")
		for btn in self.srcBtns:
			btn.setEnabled(True)
		for btn in self.resultBtns:
			btn.setEnabled(False)

	def addButton(self, text: str, onclick: Callable[[], None]
			   ) -> QPushButton:
		button = QPushButton(text, self)
		button.clicked.connect(onclick)
		self.buttonLayout.addWidget(button)
		return button

	def initUI(self):
		self.resize(1200, 800)
		self.setWindowTitle("HDR and stiching program")
		font = QFont()
		font.setPointSize(15)
		self.setFont(font)
		mainWidget = QWidget(self)
		self.setCentralWidget(mainWidget)
		mainLayout = QVBoxLayout(mainWidget)

		self.buttonLayout = QHBoxLayout()
		self.srcBtns = []
		self.srcBtns.append(self.addButton("新增圖片", self.addFile))
		self.srcBtns.append(self.addButton("重建HDR", self.makeHDR))
		self.srcBtns.append(self.addButton("組合全景圖", self.stiching))
		self.switchBtn = self.addButton("結果模式", self.switchMode)
		self.resultBtns = []
		self.resultBtns.append(self.addButton("儲存結果圖", self.saveResult))
		self.buttonLayout.setAlignment(Qt.AlignLeft)
		mainLayout.addLayout(self.buttonLayout)

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