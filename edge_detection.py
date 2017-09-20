import cv2
import numpy as np 

def blur(img, rows, cols):
    kernel_identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    kernel_3x3 = np.ones((3, 3), np.float32)/9.0
    kernel_5x5 = np.ones((5, 5), np.float32)/25.0


    cv2.namedWindow('Original',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original', rows+10,cols+10)
    cv2.imshow('Original', img)

    output = cv2.filter2D(img, -1, kernel_3x3)
    cv2.namedWindow('3x3 filtering',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('3x3 filtering', rows+10,cols+10)
    cv2.imshow('3x3 filtering', output)
    output = cv2.filter2D(img, -1, kernel_5x5)
    cv2.namedWindow('5x5 filtering',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('5x5 filtering',rows+10,cols+10)
    cv2.imshow('5x5 filtering', output)

    output = cv2.blur(img, (5,5))

    cv2.namedWindow('5x5 blurring',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('5x5 blurring', rows+10,cols+10)
    cv2.imshow('5x5 blurring', output)

    cv2.waitKey(0)

def edge_detector(img, rows, cols):
    sobel_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    canny = cv2.Canny(img, 50, 240)

    cv2.imshow('Original', img)
    cv2.imshow('Sobel horizontal', sobel_horizontal)
    cv2.imshow('Sobel vertical', sobel_vertical)
    cv2.imshow('Laplacian', laplacian)
    cv2.imshow('Canny', canny)

    cv2.waitKey(0)

def motion_blur(img, rows, cols):
    size = 5 

    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    # applying the kernel to the input image
    output = cv2.filter2D(img, -1, kernel_motion_blur)
    cv2.imshow('Motion blur', output)
    cv2.waitKey(0)

def sharpening(img, rows, cols):
    # generateing the kernels
    kernel_sharpen_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    kernel_sharpen_2 = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
    kernel_sharpen_3 = np.array([[-1, -1, -1, -1, -1],
                                [-1, 2, 2, 2, -1],
                                [-1, 2, 8, 2, -1],
                                [-1, 2, 2, 2, -1],
                                [-1, -1, -1, -1, -1]]) / 8.0
    # applying different kernels to the input image
    ouput_1 = cv2.filter2D(img, -1, kernel_sharpen_1)
    ouput_2 = cv2.filter2D(img, -1, kernel_sharpen_2)
    ouput_3 = cv2.filter2D(img, -1, kernel_sharpen_3)

    cv2.imshow('Sharpening', ouput_1)
    cv2.imshow('Excessive Sharpening', ouput_2)
    cv2.imshow('Edge Enhancement', ouput_3)

    cv2.waitKey(0)
#img = cv2.imread('./Images/samsung_sd.jpg', cv2.IMREAD_GRAYSCALE)
#rows, cols = img.shape
#edge_detector(img, rows, cols)
#blur(img, rows, cols)

img = cv2.imread('./Images/samsung_sd.jpg')
rows, cols = img.shape[:2]
cv2.imshow('Original', img)
sharpening(img, rows, cols)
#motion_blur(img, rows, cols)