import cv2 as cv
import sys
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt

in_path = "/home/pi/Desktop/log/current/"

onlyfiles = [f for f in listdir(in_path) if isfile(join(in_path, f))]

for in_fname in onlyfiles:
  print(f"processing {in_fname}")
  #  in_fname = "disturbed_with_cookie.png"

  img = cv.imread(in_path+in_fname)
  img_bw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  den = cv.fastNlMeansDenoising(img_bw, 10, 10, 7, 21)
  cv.imwrite(f"/home/pi/Desktop/log/archive/den-{in_fname}", den)

  edges = cv.Canny(img, 100, 200)
  cv.imwrite(f"/home/pi/Desktop/log/archive/edges-{in_fname}", edges)

  img_bin = cv.adaptiveThreshold(den, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
  img_bin = cv.bitwise_not(img_bin)
  cv.imwrite(f"/home/pi/Desktop/log/archive/bin-{in_fname}", img_bin)

  (cnt, hierarchy) = cv.findContours(
      img_bin.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  # cv.CHAIN_APPROX_SIMPLE
  cnt = tuple(filter(lambda x: cv.contourArea(x) > 0, cnt))

  img_contours = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  cv.drawContours(img_contours, cnt, -1, (0, 255, 0), 1)
  cv.imwrite(f"/home/pi/Desktop/log/archive/cont-{in_fname}", img_contours)

  areas = [cv.contourArea(cv.convexHull(x)) for x in cnt]

  n, bins, patches = plt.hist(areas, 50, density=True, facecolor='g', alpha=0.75)

  plt.xlabel('pixel count')
  plt.ylabel('Occurrence')
  plt.title(f'Histogram of speck area ({len(cnt)} specks)')
  plt.grid(True)
  plt.xlim(0, 1000)
  plt.yscale('log')
  plt.savefig(f"/home/pi/Desktop/log/archive/plt-{in_fname}")
  os.system(f"mv /home/pi/Desktop/log/current/{in_fname} /home/pi/Desktop/log/archive/")
