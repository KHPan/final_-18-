import cv2
import numpy as np
import itertools
print("載入模組scipy中...", end="\r")
from scipy.ndimage import maximum_filter, minimum_filter
print("載入模組scipy成功")
import pyexr
import math
from tqdm import trange
from typing import Sequence

def imread(filename: str) -> np.ndarray:
	return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)

def imwrite(filename: str, img: np.ndarray):
	cv2.imencode("." + filename.split(".")[-1], img)[1].tofile(filename)

class HDRException(Exception):
	pass

class HDRLtimeImgLengthException(HDRException):
	def __init__(self, *arg, **kwarg):
		super().__init__("HDR類別的__init__函數輸入圖片數與曝光時間陣列的大小不符", *arg, **kwarg)

class HDRToneWithoutHDRImgException(HDRException):
	def __init__(self, *arg, **kwarg):
		super().__init__("先有HDR影像才能tonemapping", *arg, **kwarg)

class HDRNoSaveTargetException(HDRException):
	def __init__(self, *arg, **kwarg):
		super().__init__("沒有要存的東西", *arg, **kwarg)

class HDR:
	def __init__(self, imgs: Sequence[np.ndarray],
			  ltimes: Sequence[int] = None, times: Sequence[float] = None,
			  is_auto_sort = False, ltstep = 1):
		self.imgs = imgs
		if is_auto_sort or (ltimes is None and times is None):
			self.imgs = sorted(self.imgs, key=np.sum)
		if ltimes is not None:
			if len(ltimes) != len(imgs):
				raise HDRLtimeImgLengthException()
			self.ltimes = ltimes
		elif times is not None:
			if len(times) != len(imgs):
				raise HDRLtimeImgLengthException()
			self.ltimes = [math.log2(t) for t in times]
		else:
			self.ltimes = [ltstep * i for i in range(0, len(imgs))]
	
	def alignment(self):
		'''圖片對齊'''
		#把所有圖片依曝光時間排序
		imgtuple = list(zip(self.ltimes, self.imgs))
		self.ltimes = []
		self.imgs = []
		for t, i in sorted(imgtuple, key=lambda obj:obj[0]):
			self.ltimes.append(t)
			self.imgs.append(i)
		
		imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			for img in self.imgs]
		resize_time = 5
		dx = [0] * (len(imgs_gray)-1)
		dy = [0] * (len(imgs_gray)-1)
		for t in range(resize_time, -1, -1):
			print(f"alignment: 縮放比1/{2**t} 計算中...", end="\r")
			scale = 0.5 ** t
			imgs_resize = [cv2.resize(img, None, fx=scale, fy=scale)
				for img in imgs_gray]
			thresholds = [np.percentile(data, [30, 70])
				for data in imgs_resize]
			dx = [x*2 for x in dx]
			dy = [y*2 for y in dy]
			for i in range(len(dx)):
				min_cnt = None
				p1 = imgs_resize[i]
				t1 = thresholds[i]
				p2 = imgs_resize[i+1]
				t2 = thresholds[i+1]
				start = (max(0, -dx[i]+1),
						 max(0, -dy[i]+1))
				end = (min(p1.shape[0], p2.shape[0]-dx[i]-1),
					   min(p1.shape[1], p2.shape[1]-dy[i]-1))
				sub_p1 = p1[start[0] : end[0], start[1] : end[1]]
				for dx0, dy0 in itertools.product(
					range(dx[i]-1, dx[i]+2), range(dy[i]-1, dy[i]+2)):
					sub_p2 = p2[start[0]+dx0 : end[0]+dx0,
								start[1]+dy0 : end[1]+dy0]
					cnt = np.count_nonzero(
						((sub_p1 <= t1[0]) & (sub_p2 >= t2[1])) | 
						((sub_p1 >= t1[1]) & (sub_p2 <= t2[0])))
					if min_cnt is None or cnt < min_cnt:
						min_cnt = cnt
						min_dx = dx0
						min_dy = dy0
				dx[i] = min_dx
				dy[i] = min_dy
			print(f"alignment: 縮放比1/{2**t} 計算完成")
		dx = [0] + list(itertools.accumulate(dx))
		mn = min(dx)
		dx = [n - mn for n in dx]
		dy = [0] + list(itertools.accumulate(dy))
		mn = min(dy)
		dy = [n - mn for n in dy]
		szx = min(p.shape[0] - d for p,d in
			zip(imgs_gray, dx))
		szy = min(p.shape[1] - d for p,d in
			zip(imgs_gray, dy))
		self.imgs = [img[dx0:dx0+szx, dy0:dy0+szy, :]
			for img, dx0, dy0 in zip(self.imgs, dx, dy)]
	
	def aladot(self, dot_num: int):
		'''回傳dot_num個適合做HDR分析的點'''
		msk_shape = 20
		print("取點中...", end="\r")

		smooth_msk = np.ones(self.imgs[0].shape[:2], dtype=bool)
		middle_msk = np.zeros(self.imgs[0].shape[:2], dtype=bool)
		for img in self.imgs:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			smooth_msk &= (maximum_filter(img, size=msk_shape) -
						   minimum_filter(img, size=msk_shape) < 50)
			middle_msk |= ((img >= 84) & (img <= 170))
		
		all_msk = smooth_msk & middle_msk
		dots = np.column_stack(np.where(all_msk))
		np.random.shuffle(dots)
		print("取點結束")
		if dots.shape[0] < dot_num:
			print(f"取點錯誤:只有{dots.shape[0]}個可選的點，無法取到{dot_num}個點")
		return dots[ : dot_num]

	def makeHDR(self, smooth_coustant: float = 100.0):
		'''重建HDR'''

		wnp = (lambda num: 1 - np.abs(num.astype(np.float64) - 128) / 129)
		w = (lambda num: float(num) / 128 if num <= 128
			else float(256-num) / 128)
		pic_num = len(self.imgs)
		dot_num = int(256 * 2 / (pic_num - 1))
		dots = self.aladot(dot_num)
		dot_num = dots.shape[0]
		hdr = np.zeros((self.imgs[0].shape[0], self.imgs[0].shape[1],
						3), dtype=np.float64)
		for clr in range(0, 3):
			print(f"構建HDR的第{clr}個顏色中...", end="\r")
			A = np.zeros((dot_num * pic_num + 255, 256 + dot_num),
						dtype=np.float64)
			B = np.zeros((dot_num * pic_num + 255,), dtype=np.float64)
			index = 0
			for img, ltime in zip(self.imgs, self.ltimes):
				for di, dot in enumerate(dots):
					value = img[*dot, clr]
					wvalue = w(value)
					A[index, value] = wvalue
					A[index, 256+di] = -wvalue
					B[index] = wvalue * ltime
					index += 1
			A[index, 127] = 10000
			B[index] = 0
			index += 1
			for value in range(1, 255):
				co = w(value) * smooth_coustant
				A[index, value-1] = co
				A[index, value] = -co * 2
				A[index, value+1] = co
				index += 1
			g_func = np.linalg.lstsq(A, B, rcond=None)[0][:256]
			div_up = np.zeros(self.imgs[0].shape[:2], dtype=np.float64)
			div_down = np.zeros(self.imgs[0].shape[:2], dtype=np.float64)
			for img, ltime in zip(self.imgs, self.ltimes):
				w_co = wnp(img[:, :, clr])
				div_down += w_co
				div_up += w_co * (g_func[img[:, :, clr]] - ltime)
			hdr[:, :, clr] = np.power(2, div_up / div_down)
			print(f"構建HDR的第{clr}個顏色完成")
		self.hdr = hdr

	def saveHDR(self, filename: str):
		if not hasattr(self, "hdr"):
			raise HDRNoSaveTargetException()
		pyexr.write(filename, self.hdr)
	
	def tonemappingL(self, a: float = 1.0,
					 b: float = 0.0, Lwhite: float = 1e10):
		if not hasattr(self, "hdr"):
			raise HDRToneWithoutHDRImgException()
		Laver = np.exp(np.sum(np.log(self.hdr + b)) / self.hdr.size)
		Lm = self.hdr * a / Laver
		Ld = (Lm / Lwhite / Lwhite + 1) * Lm / (Lm + 1)
		tone = (np.clip(Ld, 0.0, 1.0) * 255).astype(np.uint8)
		self.tone = tone
	
	def tonemappingBil(self, smooth_contant: int = 20,
					   smooth_min: int = 0, smooth_max: int = 256,
					   fre_contant: int = 256):
		if not hasattr(self, "hdr"):
			raise HDRToneWithoutHDRImgException()
		print("tonemapping...")
		inten = np.mean(self.hdr, axis=-1)
		color = self.hdr / np.expand_dims(inten, axis=-1)
		if not hasattr(self, "inten_smooth"):
			msk = np.stack(np.indices((smooth_contant*2+1,
									smooth_contant*2+1)), axis=-1)
			msk = np.sum(np.power(msk - smooth_contant, 2), axis=-1)
			msk = np.exp(-msk / (smooth_contant * smooth_contant / 2))
			msk /= smooth_contant * math.sqrt(math.pi / 2)
			div_up = np.zeros(inten.shape, dtype=np.float64)
			div_down = np.zeros(inten.shape, dtype=np.float64)
			for dx in trange(-smooth_contant, smooth_contant+1, unit="row"):
				for dy in range(-smooth_contant, smooth_contant+1):
					origin = inten[max(0, dx) : inten.shape[0] + min(0, dx),
								max(0, dy) : inten.shape[1] + min(0, dy)]
					new = inten[max(0, -dx) : inten.shape[0] + min(0, -dx),
								max(0, -dy) : inten.shape[1] + min(0, -dy)]
					co = (msk[dx+smooth_contant, dy+smooth_contant] *
						np.exp(-np.power((new - origin) / origin, 2) * 10))
					div_up[max(0, dx) : inten.shape[0] + min(0, dx),
						max(0, dy) : inten.shape[1] + min(0, dy)] += co * new
					div_down[max(0, dx) : inten.shape[0] + min(0, dx),
							max(0, dy) : inten.shape[1] + min(0, dy)] += co
			inten_smooth = div_up / div_down
			self.inten_smooth = inten_smooth
		inten_smooth = self.inten_smooth
		inten_fre = (inten - inten_smooth) / inten_smooth
		inten_smooth = np.log(inten_smooth)
		inten_smooth = ((inten_smooth - np.min(inten_smooth))
						* (smooth_max - smooth_min)
						/ (np.max(inten_smooth) - np.min(inten_smooth))
						+ smooth_min)
		inten = inten_smooth + inten_fre * fre_contant
		tone = np.clip(np.expand_dims(inten, axis=-1)
						* color, 0, 255).astype(np.uint8)
		print("tonemapping finish")
		self.tone = tone
	
	def saveTone(self, filename: str):
		if not hasattr(self, "tone"):
			raise HDRNoSaveTargetException()
		imwrite(filename, self.tone)

def makeHDR(imgs: Sequence[np.ndarray], ltimes: Sequence[int] = None,
				 times: Sequence[float] = None, align = True) -> np.ndarray:
	hdr = HDR(imgs, ltimes, times)
	if align:
		hdr.alignment()
	hdr.makeHDR()
	hdr.tonemappingBil()
	return hdr.tone