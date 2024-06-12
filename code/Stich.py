import numpy as np
import cv2
from typing import Callable, Tuple, Sequence
print("載入scipy中...", end="\r")
from scipy.signal import convolve2d
print("載入scipy成功 ")
import itertools
import math
import functools
from tqdm import tqdm, trange
import random

def imread(filename: str):
	'''讀圖檔，取代cv2.imread的功能，解決無法開啟中文檔案的問題'''
	return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)

class WarpException(Exception):
	pass

def warp(source: np.ndarray, size: tuple,
		 trans: np.ndarray | Callable[[np.ndarray, np.ndarray], 
									Tuple[np.ndarray, np.ndarray]]):
	'''注意這裡的mat是ans to src'''
	if isinstance(trans, np.ndarray):
		mat = trans
		if mat.shape != (2, 3):
			raise WarpException(f"warp的矩陣「{mat}」必須為3*2")
		def trans(*xy):
			base = np.stack((*xy, np.ones(size[::-1])), axis=0)
			return np.dot(mat, base.reshape(3, -1)).reshape(2, *size[::-1])
	ptx, pty = trans(*np.indices(size[::-1])[::-1].astype(np.float32))
	out = ((ptx < 0) | (ptx > source.shape[1] - 2) | 
		   (pty < 0) | (pty > source.shape[0] - 2))
	ptx = np.clip(ptx, 0, source.shape[1] - 2)
	pty = np.clip(pty, 0, source.shape[0] - 2)
	ptxf = np.floor(ptx).astype(np.int32)
	ptxc = ptxf + 1
	ptyf = np.floor(pty).astype(np.int32)
	ptyc = ptyf + 1
	if len(source.shape) == 2:
		expand = lambda a: a
	else:
		expand = functools.partial(np.expand_dims, axis=-1)
	picyc = (source[ptyc, ptxc] * expand(ptx - ptxf)
		   + source[ptyc, ptxf] * expand(ptxc - ptx))
	picyf = (source[ptyf, ptxc] * expand(ptx - ptxf)
		   + source[ptyf, ptxf] * expand(ptxc - ptx))
	pic = (picyc * expand(pty - ptyf) +
	 	   picyf * expand(ptyc - pty))
	pic[out] = 0
	return np.clip(pic, 0, 255).astype(np.uint8)

def getGussionMatric(GaussianSD: int):
	if not isinstance(GaussianSD, int):
		GaussianSD = int(GaussianSD)
	G = np.stack(np.indices((GaussianSD * 4 + 1,) * 2), axis=-1)
	G = np.sum(np.power(G - (GaussianSD * 2), 2), axis=-1)
	G = np.exp(-G.astype(np.float32) / 2 / GaussianSD / GaussianSD)
	G /= np.sum(G) * 3
	return G

def FDetectionHarris(pic: np.ndarray, k: float = 0.06,
		RThreshold: float = 0.5, GaussianSD: int = 4
		) -> Sequence[Tuple[int, int]]:
	'''用Harris corner detector偵測特徵點，k是R=ab-k(a+b)，RThreshold是R的閥值'''
	I = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY).astype(np.float32)
	I0 = (I[2:, 1:-1] - I[:-2, 1:-1]) / 2
	I1 = (I[1:-1, 2:] - I[1:-1, :-2]) / 2
	G = getGussionMatric(GaussianSD)
	S00 = convolve2d(I0 ** 2, G, mode="valid")
	S01 = convolve2d(I0 * I1, G, mode="valid")
	S11 = convolve2d(I1 ** 2, G, mode="valid")
	R = (S00 * S11 - S01 * S01) - k * (S00 + S11) ** 2
	msk = (R[1:-1, 1:-1] >= RThreshold)
	msk &= (R[1:-1, 1:-1] >= R[:-2, 1:-1])
	msk &= (R[1:-1, 1:-1] >= R[1:-1, :-2])
	msk &= (R[1:-1, 1:-1] >= R[2:, 1:-1])
	msk &= (R[1:-1, 1:-1] >= R[1:-1, 2:])
	msk &= (R[1:-1, 1:-1] >= R[:-2, :-2])
	msk &= (R[1:-1, 1:-1] >= R[2:, :-2])
	msk &= (R[1:-1, 1:-1] >= R[2:, 2:])
	msk &= (R[1:-1, 1:-1] >= R[:-2, 2:])
	dots = np.column_stack(np.where(msk)) + GaussianSD * 2 + 2
	return dots

def warpToCylinder(pic: np.ndarray, focalLength: int = 500
		)->Tuple[np.ndarray,
			Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
	size = (int(focalLength * np.arctan(pic.shape[1] / 2 / focalLength) * 2),
		 pic.shape[0])
	def new2old(xx, yy):
		x = focalLength * np.tan((xx - size[0]/2) / 
							focalLength) + pic.shape[1]/2
		y = (yy - size[1]/2) / np.cos((xx - size[0]/2) / 
								focalLength) + pic.shape[0]/2
		return x, y
	def old2new(x, y):
		xx = focalLength * np.arctan((x - pic.shape[1]/2) / 
							   focalLength) + size[0]/2
		yy = focalLength * ((y - pic.shape[0]/2) / 
					  np.sqrt((x - pic.shape[1]/2)**2 + 
			   focalLength**2)) + size[1]/2
		return xx, yy
	return warp(pic, size, new2old), old2new

def FDescriptionNaive(pic: np.ndarray, dots: np.ndarray,
			)->Tuple[np.ndarray, np.ndarray]:
	dots = dots[(dots[:, 0] > 1) & (dots[:, 0] < pic.shape[0]-2) &
			    (dots[:, 1] > 1) & (dots[:, 1] < pic.shape[1]-2)]
	ans = np.ones((dots.shape[0], 5, 5, 3))
	for dx, dy in itertools.product(range(-2, 3), range(-2, 3)):
		ans[:, dx, dy] = pic[dots[:, 0] + dx, dots[:, 1] + dy]
	return dots, ans.reshape((dots.shape[0], -1))

def FDescriptionSIFT(pic: np.ndarray, dots: np.ndarray,
			s_default: int = 6)->Tuple[np.ndarray, np.ndarray]:
	dots = dots[(dots[:, 0] > 1) & (dots[:, 0] < pic.shape[0]-2) &
			    (dots[:, 1] > 1) & (dots[:, 1] < pic.shape[1]-2)]
	pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY).astype(np.int32)
	d0 = pic[2:, 1:-1] - pic[:-2, 1:-1]
	d1 = pic[1:-1, 2:] - pic[1:-1, :-2]
	m = np.sqrt(d0**2 + d1**2)
	theta = np.arctan2(d0, d1)
	theta = theta * 36 / 2 / np.pi
	theta = np.floor(theta).astype(np.int32)
	new_dots = []
	orient = []
	if dots.shape[-1] == 2:
		default_G = getGussionMatric(s_default * 1.5)
	for dot in dots-1:
		if dots.shape[-1] == 2:
			G = default_G
		else:
			G = getGussionMatric(dot[2] * 1.5)
		radius = ((np.array(G.shape) - 1) / 2).astype(np.int32)
		corner1 = dot - radius
		if corner1[0] < 0:
			G = G[(-corner1[0]):, :]
			corner1[0] = 0
		if corner1[1] < 0:
			G = G[:, (-corner1[1]):]
			corner1[1] = 0
		corner2 = dot + radius + 1
		if corner2[0] > m.shape[0]:
			G = G[:(m.shape[0] - corner2[0]), :]
			corner2[0] = m.shape[0]
		if corner2[1] > m.shape[1]:
			G = G[:, :(m.shape[1] - corner2[1])]
			corner2[1] = m.shape[1]
		vote = m[corner1[0] : corner2[0], corner1[1] : corner2[1]]
		vote *= G
		candidate = theta[corner1[0] : corner2[0],
						  corner1[1] : corner2[1]]
		result = np.zeros((36, ), dtype=np.float32)
		for v, c in zip(vote.reshape(-1), candidate.reshape(-1)):
			result[c] += v
		step = math.pi * 2 / 36
		for o in np.where(result >= np.max(result) * 0.8)[0]:
			new_dots.append(dot+1)
			orient.append(step * o + step / 2)
	descs = []
	for dot, ori in zip(new_dots, orient):
		mat = np.array([[math.cos(ori), -math.sin(ori), 0],
				  		[math.sin(ori), math.cos(ori), 0]])
		non_move = np.dot(mat, [8, 8, 1])
		mat[:, 2] = [dot[1], dot[0]] - non_move
		patch = warp(pic, (17,17), mat).astype(np.int32)
		d0 = patch[1:, :-1] - patch[:-1, :-1]
		d1 = patch[:-1, 1:] - patch[:-1, :-1]
		m = np.sqrt(d0**2 + d1**2) * getGussionMatric(4)[:-1, :-1]
		theta = np.arctan2(d0, d1)
		theta = theta * 8 / 2 / np.pi
		theta = np.floor(theta).astype(np.int32)
		desc = np.zeros((4, 4, 8))
		for i0, i1 in itertools.product(range(0, 16), range(0, 16)):
			desc[int(i0 / 4), int(i1 / 4), theta[i0, i1]] += m[i0, i1]
		descs.append(desc.reshape(-1))
	return np.array(new_dots), np.array(descs)

def transDots(dots: np.ndarray,
		transfunc: Callable[[np.ndarray, np.ndarray],
					  Tuple[np.ndarray, np.ndarray]])->np.ndarray:
	return np.column_stack(transfunc(dots[:, 1].astype(np.float32),
						dots[:, 0].astype(np.float32))[::-1]).astype(np.int32)

def getMin(numbers: np.ndarray, first2second: float)->int:
	first = np.argmin(numbers)
	first_dis = numbers[first]
	numbers[first] = np.inf
	second = np.argmin(numbers)
	second_dis = numbers[second]
	if second_dis > first_dis * first2second:
		return first
	else:
		return -1

def FMatch(dots1: np.ndarray, desc1: np.ndarray,
				   dots2: np.ndarray, desc2: np.ndarray,
				   first2second: float,
				   func: Callable[[np.ndarray, np.ndarray],
					  np.ndarray])->np.ndarray:
	ans = []
	for index1 in range(dots1.shape[0]):
		index2 = getMin(func(desc2, desc1[index1]), first2second)
		if index2 != -1:
			if getMin(func(desc1, desc2[index2]),
			 			first2second) == index1:
				ans.append([dots1[index1], dots2[index2]])
	return np.array(ans)

def distance(a: np.ndarray, b: np.ndarray)->np.ndarray:
	return np.linalg.norm(a - b, axis=-1)

def FMatchDistance(dots1: np.ndarray, desc1: np.ndarray,
				   dots2: np.ndarray, desc2: np.ndarray,
				   first2second: float = 1.1)->np.ndarray:
	return FMatch(dots1, desc1, dots2, desc2, first2second, distance)

def angle(a: np.ndarray, b: np.ndarray)->np.ndarray:
	dot = np.sum(a * b, axis=-1)
	a_ = np.linalg.norm(a, axis=-1)
	a_[a_ == 0] = 1
	b_ = np.linalg.norm(b, axis=-1)
	if b_ == 0:
		b_ = 1
	return np.arccos(dot / a_ / b_)

def FMatchAngle(dots1: np.ndarray, desc1: np.ndarray,
				dots2: np.ndarray, desc2: np.ndarray,
				first2second: float = 1.1)->np.ndarray:
	return FMatch(dots1, desc1, dots2, desc2, first2second, angle)

def FitRANSAC(matches: np.ndarray, k_times: int = 10000,
			  n_sample: int = 6, deviation_threshold: float = 15.0
			  )->np.ndarray:
	if not isinstance(matches, np.ndarray):
		matches = np.array(matches)
	max_valid = 0
	valid = None
	for _ in trange(0, k_times, desc="RANSAC"):
		sample = random.sample(range(0, matches.shape[0]), n_sample)
		A = np.zeros((n_sample * 2, 6), dtype=np.float32)
		A[::2, :2] = matches[sample, 0]
		A[::2, 2] = 1
		A[1::2, 3:5] = matches[sample, 0]
		A[1::2, 5] = 1
		B = matches[sample, 1].reshape(-1)
		mat = np.linalg.lstsq(A, B, rcond=None)[0].reshape((2, 3))
		old_dots = np.ones((3, matches.shape[0]), dtype=np.float32)
		old_dots[:2] = matches[:, 0].reshape((-1, 2)).T
		cal = np.dot(mat, old_dots)
		D = np.linalg.norm(cal.T - matches[:, 1], axis=-1)
		vld = (D <= deviation_threshold)
		vld_num = np.sum(vld)
		if vld_num > max_valid:
			max_valid = vld_num
			valid = vld
	A = np.zeros((max_valid * 2, 6), dtype=np.float32)
	A[::2, :2] = matches[valid, 0]
	A[::2, 2] = 1
	A[1::2, 3:5] = matches[valid, 0]
	A[1::2, 5] = 1
	B = matches[valid, 1].reshape(-1)
	mat = np.linalg.lstsq(A, B, rcond=None)[0].reshape((2, 3))
	return mat

def FitRANSAC2(matches: np.ndarray, k_times: int = 10000,
			  n_sample: int = 6, deviation_threshold: float = 15.0
			  )->np.ndarray:
	if not isinstance(matches, np.ndarray):
		matches = np.array(matches)
	max_valid = 0
	valid = None
	for _ in trange(0, k_times, desc="RANSAC"):
		sample = random.sample(range(0, matches.shape[0]), n_sample)
		center = np.mean(matches[sample, 1] - matches[sample, 0], axis=0)
		D = np.linalg.norm(center + matches[:, 0] - matches[:, 1], axis=-1)
		vld = (D <= deviation_threshold)
		vld_num = np.sum(vld)
		if vld_num > max_valid:
			max_valid = vld_num
			valid = vld
	center = np.mean(matches[valid, 1] - matches[valid, 0], axis=0)
	mat = np.zeros((2, 3), dtype=np.float32)
	mat[0, 0] = 1
	mat[1, 1] = 1
	mat[:, 2] = center
	return mat

def Blending(pic1: np.ndarray, pic2: np.ndarray, mat: np.ndarray
			 )->np.ndarray:
	mat = mat.copy()
	inv_mat = -mat
	inv_mat[:, :2] = np.linalg.inv(mat[:, :2])
	corners = np.zeros((3, 4), dtype=np.float32)
	corners[2] = 1
	corners[0, :2] = pic2.shape[0]
	corners[1, 1:3] = pic2.shape[1]
	corners = np.dot(inv_mat, corners).astype(np.int32)
	pic2corner1 = np.min(corners, axis=1)
	pic2corner2 = np.max(corners, axis=1)
	pic1corner1 = np.array([0, 0])
	pic1corner2 = np.array(pic1.shape[:2])
	all_corner1 = np.min((pic1corner1, pic2corner1), axis=0)
	all_corner2 = np.max((pic1corner2, pic2corner2), axis=0)
	mat[:, 2] += np.dot(mat[:, :2], all_corner1 - pic1corner1)
	pic1corner1 -= all_corner1
	pic2corner1 -= all_corner1
	pic1corner2 -= all_corner1
	pic2corner2 -= all_corner1
	all_corner2 -= all_corner1
	mat[:, :2] = mat[:, 1::-1]
	mat = mat[::-1]
	new_pic = warp(pic2, all_corner2[::-1], mat)
	pic1percent = np.ones(pic1.shape[:2], dtype=np.float32)
	cover_corner1 = np.max((pic1corner1, pic2corner1), axis=0) - pic1corner1
	cover_corner2 = np.min((pic1corner2, pic2corner2), axis=0) - pic1corner1
	if cover_corner1[1] == 0:
		pic1percent[cover_corner1[0] : cover_corner2[0],
			  		cover_corner1[1] : cover_corner2[1]] = \
			np.arange(0, cover_corner2[1] - cover_corner1[1]
			 ).astype(np.float32) / (cover_corner2[1] - cover_corner1[1])
	else:
		pic1percent[cover_corner1[0] : cover_corner2[0],
			  		cover_corner1[1] : cover_corner2[1]] = \
			np.arange(cover_corner2[1] - cover_corner1[1], 0, -1
			 ).astype(np.float32) / (cover_corner2[1] - cover_corner1[1])
	sub_new_pic = new_pic[pic1corner1[0] : pic1corner2[0],
					      pic1corner1[1] : pic1corner2[1]]
	pic1percent[(sub_new_pic[:, :, 0] == 0) &
			 	(sub_new_pic[:, :, 1] == 0) &
				(sub_new_pic[:, :, 2] == 0)] = 1
	pic1percent[(pic1[:, :, 0] == 0) &
			 	(pic1[:, :, 1] == 0) &
				(pic1[:, :, 2] == 0)] = 0
	pic1percent = np.expand_dims(pic1percent, axis=-1)
	new_pic[pic1corner1[0] : pic1corner2[0],
			pic1corner1[1] : pic1corner2[1]] = (pic1 * pic1percent +
						sub_new_pic * (1 - pic1percent))
	return new_pic, pic2corner1

def pair(it):
	it = iter(it)
	a = next(it)
	try:
		while True:
			b = next(it)
			yield (a, b)
			a = b
	except StopIteration:
		return

class StichException(Exception):
	pass

class StichDescriptionWithoutFeathuresException(StichException):
	def __init__(self, *arg, **kwarg):
		super().__init__("在做特徵點描述之前需先尋找特徵點", *arg, **kwarg)

class StichMatchWithoutDescriptionException(StichException):
	def __init__(self, *arg, **kwarg):
		super().__init__("在做特徵點比對之前需先描述特徵點", *arg, **kwarg)

class StichMatchWithoutFeathuresException(StichException):
	def __init__(self, *arg, **kwarg):
		super().__init__("在做特徵點比對之前需先有特徵點", *arg, **kwarg)

class StichFitWithoutMatchException(StichException):
	def __init__(self, *arg, **kwarg):
		super().__init__("在趨近位移矩陣之前需先做特徵點比對", *arg, **kwarg)

class Stich:
	def __init__(self, imgs: Sequence[np.ndarray]):
		self.imgs = imgs
	
	def FDetectionHarris(self, k: float = 0.06,
			RThreshold: float = 0.5, GaussianSD: int = 4):
		self.feathures = []
		for img in self.imgs:
			self.feathures.append(FDetectionHarris(
				img, k, RThreshold, GaussianSD))
	
	def warpToCylinder(self, focalLength: int = 500):
		imgcpy = self.imgs
		self.imgs = []
		if hasattr(self, "feathures"):
			fcpy = self.feathures
			self.feathures = []
			for img, dots in zip(imgcpy, fcpy):
				img, transfunc = warpToCylinder(img, focalLength)
				dots = transDots(dots, transfunc)
				self.imgs.append(img)
				self.feathures.append(dots)
		else:
			for img in imgcpy:
				img, _ = warpToCylinder(img, focalLength)
				self.imgs.append(img)
	
	def FDescriptionNaive(self):
		if not hasattr(self, "feathures"):
			raise StichDescriptionWithoutFeathuresException()
		self.descriptions = []
		fcpy = self.feathures
		self.feathures = []
		for img, dots in zip(self.imgs, fcpy):
			dots, desc = FDescriptionNaive(img, dots)
			self.feathures.append(dots)
			self.descriptions.append(desc)

	def FDescriptionSIFT(self, s_default: int = 6):
		if not hasattr(self, "feathures"):
			raise StichDescriptionWithoutFeathuresException()
		self.descriptions = []
		fcpy = self.feathures
		self.feathures = []
		for img, dots in zip(self.imgs, fcpy):
			dots, desc = FDescriptionSIFT(img, dots, s_default)
			self.feathures.append(dots)
			self.descriptions.append(desc)
	
	def FMatchDistance(self, first2second: float = 1.1):
		if not hasattr(self, "feathures"):
			raise StichMatchWithoutFeathuresException()
		if not hasattr(self, "description"):
			raise StichMatchWithoutDescriptionException()
		self.matches = []
		for (dots1, desc1), (dots2, desc2) in pair(zip(self.feathures,
													self.descriptions)):
			mtch = FMatchDistance(dots1, desc1, dots2, desc2, first2second)
			self.matches.append(mtch)

	def FMatchAngle(self, first2second: float = 1.1):
		if not hasattr(self, "feathures"):
			raise StichMatchWithoutFeathuresException()
		if not hasattr(self, "description"):
			raise StichMatchWithoutDescriptionException()
		self.matches = []
		for (dots1, desc1), (dots2, desc2) in pair(zip(self.feathures,
													self.descriptions)):
			mtch = FMatchAngle(dots1, desc1, dots2, desc2, first2second)
			self.matches.append(mtch)
	
	def FitRANSAC(self, k_times: int = 10000,
			n_sample: int = 6, deviation_threshold: float = 15.0):
		if not hasattr(self, "matches"):
			raise StichFitWithoutMatchException()
		self.fit_mats = []
		for mtch in self.matches:
			mat = FitRANSAC(mtch, k_times, n_sample, deviation_threshold)
			self.fit_mats.append(mat)
	
	def FitRANSAC2(self, k_times: int = 10000,
			  n_sample: int = 6, deviation_threshold: float = 15.0):
		if not hasattr(self, "matches"):
			raise StichFitWithoutMatchException()
		self.fit_mats = []
		for mtch in self.matches:
			mat = FitRANSAC2(mtch, k_times, n_sample, deviation_threshold)
			self.fit_mats.append(mat)
	
	def Blending(self) -> np.ndarray:
		base = np.array([0, 0])
		self.result