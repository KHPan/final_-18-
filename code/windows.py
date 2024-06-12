import HDR
import cv2
if __name__ == "__main__":
	img1 = HDR.imread(r"C:\Users\hank9\OneDrive\文件\GitHub\hw1_-18-\data\PPT範例亮圖.png")
	img2 = HDR.imread(r"C:\Users\hank9\OneDrive\文件\GitHub\hw1_-18-\data\PPT範例暗圖.png")
	result = HDR.combineImage([img1, img2])
	cv2.imshow("result", result)
	cv2.waitKey(0)