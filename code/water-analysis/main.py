import cv2 as cv
import sys

import matplotlib.pyplot as plt

# cam = cv.VideoCapture(0)
# retval, frame = cam.read()
# if retval != True:
#    raise ValueError("Can't read frame")
in_path = "../../documentation/resources/"
in_fname = "disturbed_with_cookie.png"

img = cv.imread(in_path+in_fname)
img_bw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
den = cv.fastNlMeansDenoising(img_bw, 10, 10, 7, 21)
cv.imwrite("out/den.png", den)

edges = cv.Canny(img, 100, 200)
cv.imwrite(f"out/out_img_edges.png", edges)

img_bin = cv.adaptiveThreshold(den, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
img_bin = cv.bitwise_not(img_bin)
cv.imwrite(f"out/out_img_bin.png", img_bin)

(cnt, hierarchy) = cv.findContours(
    img_bin.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  # cv.CHAIN_APPROX_SIMPLE
cnt = tuple(filter(lambda x: cv.contourArea(x) > 0, cnt))

img_contours = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.drawContours(img_contours, cnt, -1, (0, 255, 0), 1)
cv.imwrite("out/cont.png", img_contours)

areas = [cv.contourArea(cv.convexHull(x)) for x in cnt]

n, bins, patches = plt.hist(areas, 50, density=True, facecolor='g', alpha=0.75)

plt.xlabel('pixel count')
plt.ylabel('Occurrence')
plt.title(f'Histogram of speck area ({len(cnt)} specks)')
plt.grid(True)
plt.xlim(0, 1000)
plt.yscale('log')
plt.savefig(f"out/plt_{in_fname}")
plt.show()


