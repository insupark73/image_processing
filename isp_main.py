import cv2
import numpy as np 
import filter_operation as filter

#img = cv2.imread('./Images/samsung_sd.jpg', cv2.IMREAD_GRAYSCALE)
#rows, cols = img.shape
#edge_detector(img, rows, cols)
#blur(img, rows, cols)

img = cv2.imread('./Images/snow.jpg')
rows, cols = img.shape[:2]
cv2.imshow('Original', img)
#sharpening(img, rows, cols)
#motion_blur(img, rows, cols)
#embossing(img, rows, cols)
#morphology(img, rows, cols)
filter.vignette(img, rows, cols)
#filter.histo_equal_color(img, rows, cols)