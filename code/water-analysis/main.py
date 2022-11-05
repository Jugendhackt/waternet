import cv2 as cv
import sys

# cam = cv.VideoCapture(0)
# retval, frame = cam.read()
# if retval != True:
#    raise ValueError("Can't read frame")


img_bw = cv.imread("/home/frederic/Desktop/test.jpg", cv.IMREAD_GRAYSCALE)
(thresh, img_bin) = cv.threshold(img_bw, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
print(thresh)

th2 = cv.adaptiveThreshold(img_bw, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
th3 = cv.adaptiveThreshold(img_bw, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

cv.imwrite(f"out/output.png", img_bin)
cv.imwrite(f"out/output-ad-mean.png", th2)
cv.imwrite(f"out/output-ad-gaussian.png", th3)

dst = cv.fastNlMeansDenoising(img_bw, 10, 10, 7, 21)


(thresh, img_bin) = cv.threshold(dst, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
print(thresh)

th2 = cv.adaptiveThreshold(dst, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
th3 = cv.adaptiveThreshold(dst, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

cv.imwrite(f"out/output-den.png", img_bin)
cv.imwrite(f"out/output-ad-mean-den.png", th2)
cv.imwrite(f"out/output-ad-gaussian-den.png", th3)
cv.imwrite("out/den.png", dst)
