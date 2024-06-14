import HDR
import Stich
import cv2
import sys
from PyQt5.QtWidgets import (QWidget, QMainWindow, QApplication,
	QVBoxLayout, QHBoxLayout, QFormLayout, QFrame, QScrollArea,
	QLabel, QPushButton, QComboBox, QLineEdit, QCheckBox,
	QMenu, QAction,
	QFileDialog, QMessageBox, QDialog, QFontDialog)
from PyQt5.QtGui import (QFont, QPixmap, QImage,
	QDoubleValidator, QIntValidator)
from PyQt5.QtCore import Qt
import numpy as np
from typing import Callable, Tuple, Sequence
import inspect

def getLine() -> QFrame:
	line = QFrame()
	line.setFrameShape(QFrame.HLine)
	line.setFrameShadow(QFrame.Sunken)
	return line

meaning = {"fre_constant": "細節對比", "smooth_constant":"mask大小平滑", "smooth_coustant": "平滑",
		    "smooth_min":"亮度mapping最小值", "smooth_max":"亮度mapping最大值", "first2second": "配一到二"
			, "s_default": "SIFT基礎", "k_times": "what",
			"focalLength": "焦距[單位像素]", "k": "KKK", "RThreshold":"Response值門檻", "GaussianSD":"高斯",
			"k": "小k", "n_sample": "取樣", "deviation_threshold":"dth"}

class FuncForm:
	def __init__(self, func: Sequence[Callable] | Callable,
			  parent):
		self.parent = parent
		self.origin_index = self.index = 0
		if not isinstance(func, list):
			func = [func]
		self.func = func
		self.value = [[] for _ in range(len(func))]
		self.type = [[] for _ in range(len(func))]
		self.key = [[] for _ in range(len(func))]
		for origin, func, type, key in zip(self.value, self.func,
										self.type, self.key):
			for name, param in inspect.signature(func).parameters.items():
				if param.default == inspect.Parameter.empty:
					continue
				origin.append(param.default)
				type.append(param.annotation)
				key.append(name)
	
	def oncombo(self, index: int):
		if index != self.index:
			self.index = index
			self.setForm()

	def draw(self, mainLayout: QVBoxLayout):
		if len(self.func) == 1:
			if self.func[0].__name__ in meaning:
				mainLayout.addWidget(QLabel(meaning[self.func[0].__name__]))
			else:
				mainLayout.addWidget(QLabel(self.func[0].__name__))
		else:
			combo = QComboBox(self.parent)
			for func in self.func:
				if func.__name__ in meaning:
					combo.addItem(meaning[func.__name__])
				else:
					combo.addItem(func.__name__)
			combo.activated.connect(self.oncombo)
			mainLayout.addWidget(combo)
		self.form = QFormLayout()
		self.setForm()
		mainLayout.addLayout(self.form)
	
	def setForm(self):
		while self.form.count():
			self.form.takeAt(0).widget().deleteLater()
		self.attrs = []
		for key, value, typ in zip(self.key[self.index],
					self.value[self.index], self.type[self.index]):
			text = QLineEdit(str(value), self.parent)
			if typ == int:
				text.setValidator(QIntValidator())
			else:
				text.setValidator(QDoubleValidator())
			if key in meaning:
				self.form.addRow(meaning[key], text)
			else:
				self.form.addRow(key, text)
			self.attrs.append(text)
	
	def valid(self) -> bool:
		for attr in self.attrs:
			if attr.text() == "":
				return False
		return True

	def modify(self) -> bool:
		if self.origin_index != self.index:
			return True
		for typ, value, attr in zip(self.type[self.index],
					self.value[self.index], self.attrs):
			if typ(attr.text()) != value:
				return True
		return False
	
	def run(self):
		run_attrs = {}
		self.origin_index = self.index
		self.value[self.index] = []
		for key, attr, typ in zip(self.key[self.index], self.attrs,
									self.type[self.index]):
			val = typ(attr.text())
			run_attrs[key] = val
			self.value[self.index].append(val)
		print(run_attrs)
		self.func[self.index](**run_attrs)

def cmp_list(lst1: Sequence, lst2: Sequence) -> bool:
	if len(lst1) != len(lst2):
		return False
	return all(i1 is i2 for i1, i2 in zip(lst1, lst2))

class HDRDialog(QDialog):
	def __init__(self, parent):
		super().__init__(parent)
		self.hdr = HDR.HDR(parent.imgs[:])
		self.funcForms = [
			FuncForm(self.hdr.makeHDR, self),
			FuncForm([self.hdr.tonemappingBil, self.hdr.tonemappingL],
				self)
		]
		self.cpy_imgs = parent.imgs[:]
		self.origin_align = True
		self.start = 0
		self.times = None
		self.initUI()

	def timeValid(self) -> bool:
		return self.autoLight.isChecked() or all(text.text() != "" 
										   for text in self.timeText)

	def timeModify(self):
		if self.times is None:
			return self.autoLight.isChecked() == False
		if self.autoLight.isChecked() == True:
			return True
		return any(1 / float(text.text()) != time
			 for text, time in zip(self.timeText, self.times))

	def setStart(self, to: int):
		if self.start > 0 and (not cmp_list(self.cpy_imgs, self.parent().imgs)
				or self.alignment.isChecked() != self.origin_align
				or self.timeModify()):
			self.start = 0
		if self.start > 1 and self.funcForms[0].modify():
			self.start = 1
		if to <= 2:
			return
		if self.start > 2 and self.funcForms[1].modify():
			self.start = 2

	def runModify(self, to: int):
		self.setStart(to)
		print("start:", self.start)
		if self.start <= 0:
			if self.autoLight.isChecked():
				self.times = None
			else:
				self.times = []
				for text in self.timeText:
					self.times.append(1 / float(text.text()))
			self.hdr.__init__(self.parent().imgs[:], times=self.times)
			self.cpy_imgs = self.parent().imgs[:]
			self.origin_align = self.alignment.isChecked()
			if self.origin_align:
				self.hdr.alignment()
			else:
				sz0 = min(img.shape[0] for img in self.hdr.imgs)
				sz1 = min(img.shape[1] for img in self.hdr.imgs)
				self.hdr.imgs = [img[:sz0, :sz1] for img in self.hdr.imgs]
			self.start = 1
		if self.start <= 1:
			self.funcForms[0].run()
			self.start = 2
		if self.start >= to:
			return
		if self.start <= 2:
			self.funcForms[1].run()
			self.start = 3
	
	def autoCheck(self, checked: bool):
		for text in self.timeText:
			text.setEnabled(not checked)
		if checked:
			self.times = None
		else:
			self.times = []
			for text in self.timeText:
				text.setText("1")
				self.times.append(1.0)

	def saveHDR(self):
		if not (all(funcForm.valid() for funcForm in self.funcForms)
		  		and self.timeValid()):
			QMessageBox.warning(self, "警告", "文字方塊不得為空")
			return
		
		self.runModify(2)
		filename, _ = QFileDialog.getSaveFileName(self, filter="EXR (*.exr)")
		if filename:
			self.hdr.saveHDR(filename)

	def runAll(self):
		if not (all(funcForm.valid() for funcForm in self.funcForms)
		  		and self.timeValid()):
			QMessageBox.warning(self, "警告", "文字方塊不得為空")
			return
		
		self.runModify(3)
		parent = self.parent()
		parent.result = self.hdr.tone
		parent.resultMode()

	def initUI(self):
		self.setWindowTitle("HDR Detail")
		self.setFont(self.parent().font)
		mainLayout = QVBoxLayout()

		self.autoLight = QCheckBox("自動曝光時間")
		self.autoLight.setChecked(True)
		self.autoLight.clicked.connect(self.autoCheck)
		mainLayout.addWidget(self.autoLight)

		self.timeLayout = QHBoxLayout()
		self.timeText = []
		for i in range(len(self.cpy_imgs)):
			if i > 0:
				self.timeLayout.addStretch()
			self.timeLayout.addWidget(QLabel(" 1/"))
			text = QLineEdit("1", self)
			text.setValidator(QDoubleValidator())
			text.setEnabled(False)
			self.timeText.append(text)
			self.timeLayout.addWidget(text)
		mainLayout.addLayout(self.timeLayout)

		mainLayout.addWidget(getLine())

		if "alignment" in meaning:
			self.alignment = QCheckBox(meaning["alignment"])
		else:
			self.alignment = QCheckBox("alignment")
		self.alignment.setChecked(True)
		mainLayout.addWidget(self.alignment)

		mainLayout.addWidget(getLine())

		self.funcForms[0].draw(mainLayout)
		button = QPushButton("儲存HDR檔案", self)
		button.clicked.connect(self.saveHDR)
		mainLayout.addWidget(button)

		mainLayout.addWidget(getLine())
		
		self.funcForms[1].draw(mainLayout)
		button = QPushButton("run", self)
		button.clicked.connect(self.runAll)
		mainLayout.addWidget(button)

		self.setLayout(mainLayout)
			
	def closeEvent(self, event):
		if self.parent().result is None:
			self.parent().sourceMode()
		else:
			self.parent().resultMode()
		super().closeEvent(event)

class StichDialog(QDialog):
	def __init__(self, parent):
		super().__init__(parent)
		self.stich = Stich.Stich(parent.imgs[:])
		self.funcForms = [
			FuncForm(self.stich.FDetectionHarris, self),
			FuncForm(self.stich.warpToCylinder, self),
			FuncForm([self.stich.FDescriptionSIFT,
					self.stich.FDescriptionNaive], self),
			FuncForm([self.stich.FMatchAngle,
			 		self.stich.FMatchDistance], self),
			FuncForm([self.stich.FitRANSAC2,
			 		self.stich.FitRANSAC], self)
		]
		self.cpy_imgs = parent.imgs[:]
		self.start = 0
		self.initUI()
	
	def setStart(self, to: int):
		if self.start > 0 and (not cmp_list(self.cpy_imgs, self.parent().imgs) or
				self.funcForms[0].modify() or self.funcForms[1].modify()):
			self.start = 0
		for i in range(1, 4):
			if to <= i:
				return
			if self.start > i and self.funcForms[i+1].modify():
				self.start = i
	
	def runModify(self, to: int):
		self.setStart(to)
		print("start: ", self.start)
		if self.start <= 0:
			self.stich.__init__(self.parent().imgs[:])
			self.cpy_imgs = self.parent().imgs[:]
			self.funcForms[0].run()
			self.funcForms[1].run()
			self.start = 1
		for i in range(1, 4):
			if self.start >= to:
				return
			if self.start <= i:
				self.funcForms[i+1].run()
				self.start = i+1
		self.stich.Blending()

	def displaySource(self):
		self.parent().detailMode()

	def drawFeathure(self):
		if not all(funcForm.valid() for funcForm in self.funcForms):
			QMessageBox.warning(self, "警告", "文字方塊不得為空")
			return
		
		self.runModify(1)
		demos = self.stich.drawFeathures()
		self.parent().demoMode(demos)
	
	def drawMatch(self):
		if not all(funcForm.valid() for funcForm in self.funcForms):
			QMessageBox.warning(self, "警告", "文字方塊不得為空")
			return
		
		self.runModify(3)
		self.parent().demoMode(self.stich.drawMatch())
	
	def drawFit(self):
		if not all(funcForm.valid() for funcForm in self.funcForms):
			QMessageBox.warning(self, "警告", "文字方塊不得為空")
			return
		
		self.runModify(4)
		self.parent().demoMode(self.stich.drawFit())
	
	def runAll(self):
		if not all(funcForm.valid() for funcForm in self.funcForms):
			QMessageBox.warning(self, "警告", "文字方塊不得為空")
			return
		
		self.runModify(4)
		self.parent().result = self.stich.result
		self.parent().resultMode()
	
	def addButton(self, text: str, clicked: Callable) -> QPushButton:
		button = QPushButton(text, self)
		button.clicked.connect(clicked)
		return button
	
	def initUI(self):
		self.setWindowTitle("Stich Detail")
		self.setFont(self.parent().font)
		mainLayout = QVBoxLayout()
		
		mainLayout.addWidget(self.addButton("顯示源圖", self.displaySource))
		mainLayout.addWidget(getLine())

		self.funcForms[0].draw(mainLayout)
		mainLayout.addWidget(getLine())

		self.funcForms[1].draw(mainLayout)
		mainLayout.addWidget(self.addButton("顯示特徵點", self.drawFeathure))
		mainLayout.addWidget(getLine())

		self.funcForms[2].draw(mainLayout)
		mainLayout.addWidget(getLine())
		
		self.funcForms[3].draw(mainLayout)
		mainLayout.addWidget(self.addButton("顯示特徵點比對", self.drawMatch))
		mainLayout.addWidget(getLine())

		self.funcForms[4].draw(mainLayout)
		mainLayout.addWidget(self.addButton("顯示配對位置", self.drawFit))
		mainLayout.addWidget(getLine())

		mainLayout.addWidget(self.addButton("run all", self.runAll))

		self.setLayout(mainLayout)
	
	def closeEvent(self, event):
		if self.parent().result is None:
			self.parent().sourceMode()
		else:
			self.parent().resultMode()
		super().closeEvent(event)

class PicLabel(QLabel):
	def __init__(self, parent, index: int):
		super().__init__(parent)
		self.imgs = parent.imgs
		self.update = parent.sourceMode
		self.index = index

	def addAct(self, text: str, triggered: Callable[[], None]):
		action = QAction(text, self)
		action.triggered.connect(triggered)
		self.menu.addAction(action)
	
	def contextMenuEvent(self, event):
		self.menu = QMenu(self)
		self.addAct("移除", self.onremove)
		if self.index > 0:
			self.addAct("左移", self.onleft)
		if self.index < len(self.imgs)-1:
			self.addAct("右移", self.onright)
		self.menu.exec_(event.globalPos())

	def onremove(self):
		del self.imgs[self.index]
		self.update()
	
	def onleft(self):
		self.imgs[self.index-1], self.imgs[self.index] = \
			self.imgs[self.index], self.imgs[self.index-1]
		self.update()
	
	def onright(self):
		self.imgs[self.index+1], self.imgs[self.index] = \
			self.imgs[self.index], self.imgs[self.index+1]
		self.update()

class DemoLabel(QLabel):
	def __init__(self, parent, img: np.ndarray):
		super().__init__(parent)
		self.img = img
	
	def saveDemo(self):
		filename, _ = QFileDialog.getSaveFileName(self,
						filter="PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)")
		if filename:
			HDR.imwrite(filename, self.img)

	def contextMenuEvent(self, event):
		menu = QMenu(self)
		action = QAction("save demo", self)
		action.triggered.connect(self.saveDemo)
		menu.addAction(action)
		menu.exec_(event.globalPos())

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
					self.scrollArea.size().height() - 70,
					Qt.SmoothTransformation)

	def addPic(self, img: np.ndarray, index: int = -1):
		if index == -1:
			label = QLabel(self)
		elif index == -2:
			label = DemoLabel(self, img)
		else:
			label = PicLabel(self, index)
		label.setPixmap(self.loadPixmap(img))
		self.picLayout.addWidget(label)

	def addFile(self):
		filenames, _ = QFileDialog.getOpenFileNames(self,
										filter="圖片 (*.png *.bmp *.jpg)")
		for filename in filenames:
			img = HDR.imread(filename)
			self.addPic(img, len(self.imgs))
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
		for index, img in enumerate(self.imgs):
			self.addPic(img, index)
		self.switchBtn.setEnabled(self.result is not None)
		self.switchBtn.setText("結果模式")
		for btn in self.srcBtns:
			btn.setEnabled(True)
		for btn in self.resultBtns:
			btn.setEnabled(False)
	
	def demoMode(self, demos: Sequence[np.ndarray]):
		while self.picLayout.count():
			self.picLayout.takeAt(0).widget().deleteLater()
		for demo in demos:
			self.addPic(demo, -2)
		self.switchBtn.setEnabled(False)
		for btn in self.srcBtns:
			btn.setEnabled(False)
		for btn in self.resultBtns:
			btn.setEnabled(False)
	
	def detailMode(self):
		self.sourceMode()
		self.switchBtn.setEnabled(False)
		for btn in self.srcBtns[1:]:
			btn.setEnabled(False)

	def HDRDetail(self):
		if len(self.imgs) <= 1:
			QMessageBox.warning(self, "警告", "兩個以上的圖片才能製造HDR影像")
			return
		self.detailMode()
		HDRDialog(self).show()

	def StichDetail(self):
		if len(self.imgs) <= 1:
			QMessageBox.warning(self, "警告", "兩個以上的圖片才能組合全景圖")
			return
		self.detailMode()
		StichDialog(self).show()
	
	def modifyFont(self):
		font, ok = QFontDialog.getFont(self.font, self)
		if ok:
			self.font = font
			self.setFont(self.font)

	def addButton(self, text: str, onclick: Callable[[], None]
			   ) -> QPushButton:
		button = QPushButton(text, self)
		button.clicked.connect(onclick)
		self.buttonLayout.addWidget(button)
		return button

	def initUI(self):
		self.resize(1200, 800)
		self.setWindowTitle("HDR and stiching program")
		self.font = QFont()
		self.font.setPointSize(15)
		self.setFont(self.font)
		mainWidget = QWidget(self)
		self.setCentralWidget(mainWidget)
		mainLayout = QVBoxLayout(mainWidget)

		self.buttonLayout = QHBoxLayout()
		self.srcBtns = []
		self.srcBtns.append(self.addButton("新增圖片", self.addFile))
		self.srcBtns.append(self.addButton("重建HDR", self.makeHDR))
		self.srcBtns.append(self.addButton("...", self.HDRDetail))
		self.srcBtns.append(self.addButton("組合全景圖", self.stiching))
		self.srcBtns.append(self.addButton("...", self.StichDetail))
		self.switchBtn = self.addButton("結果模式", self.switchMode)
		self.resultBtns = []
		self.resultBtns.append(self.addButton("儲存結果圖", self.saveResult))
		self.addButton("修改字型", self.modifyFont)
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