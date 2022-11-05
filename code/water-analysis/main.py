import cv2 as cv
import sys

# cam = cv.VideoCapture(0)
# retval, frame = cam.read()
# if retval != True:
#    raise ValueError("Can't read frame")

img = cv.imread("../../documentation/resources/test.jpg")
img_bw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
den = cv.fastNlMeansDenoising(img_bw, 10, 10, 7, 21)
cv.imwrite("out/den.png", den)

img_bin = cv.adaptiveThreshold(den, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
img_bin = cv.bitwise_not(img_bin)
cv.imwrite(f"out/out_img_bin.png", img_bin)

(cnt, hierarchy) = cv.findContours(
    img_bin.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
img_contours = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.drawContours(img_contours, cnt, -1, (0, 255, 0), 1)
cv.imwrite("out/cont.png", img_contours)
