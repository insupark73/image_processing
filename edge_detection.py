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
def embossing(img, rows, cols):
    # generating the kernels
    kernel_emboss_1 = np.array([[0, -1, -1],
                                [1, 0, -1],
                                [1, 1, 0]])
    kernel_emboss_2 = np.array([[-1, -1, 0],
                                [-1, 0, 1],
                                [0, 1, 1]])
    kernel_emboss_3 = np.array([[1, 0, 0],
                                [0, 0, 0],
                                [0, 0, -1]]) # converting the image to grayscale gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # applying the kernels to the grayscale image and adding the offset output_1 = cv2.filter2D(gray_img, -1, kernel_emboss_1) + 128 output_2 = cv2.filter2D(gray_img, -1, kernel_emboss_2) + 128 output_3 = cv2.filter2D(gray_img, -1, kernel_emboss_3) + 128 cv2.imshow('Embossing - South West', output_1)
    cv2.imshow('Embossing - South East', output_2)
    cv2.imshow('Embossing - North West', output_3)

    cv2.waitKey(0)

def morphology(img, rows, cols):
    kernel = np.ones((2,2), np.uint8)

    img_erosion = cv2.erode(img, kernel, iterations=1)
    img_dilation = cv2.dilate(img, kernel, iterations=1)

    cv2.imshow('Erosion', img_erosion)
    cv2.imshow('Dilation', img_dilation)

    cv2.waitKey(0)

def vignette(img, rows, cols):
    # generating vignette mask using Gaussian kernels
    kernel_x = cv2.getGaussianKernel(int(1.5*cols), 200)
    kernel_y = cv2.getGaussianKernel(int(1.5*rows), 200)

    kernel = kernel_y * kernel_x.T 
    mask = 255 * kernel / np.linalg.norm(kernel)
    mask = mask[int(0.5*rows):, int(0.5*cols):]
    output = np.copy(img)

    # applying the mask to each channel in the input image
    for i in range(3):
        output[:, :, i] = output[:, :, i] * mask

    cv2.imshow('Vignette', output)
    cv2.waitKey(0)

def histo_equal(img, rows, cols):
    # equalize the histogram of the input image
    histeq = cv2.equalizeHist(img)

    cv2.imshow('Histogram equalized', histeq)
    cv2.waitKey(0)

def histo_equal_color(img, rows, cols):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    cv2.imshow('Histogram equalized', img_output)
    cv2.waitKey(0)
#img = cv2.imread('./Images/samsung_sd.jpg', cv2.IMREAD_GRAYSCALE)
#rows, cols = img.shape
#edge_detector(img, rows, cols)
#blur(img, rows, cols)

img = cv2.imread('./Images/snow.jpg', 0)
rows, cols = img.shape[:2]
cv2.imshow('Original', img)
#sharpening(img, rows, cols)
#motion_blur(img, rows, cols)
#embossing(img, rows, cols)
#morphology(img, rows, cols)
#vignette(img, rows, cols)
histo_equal_color(img, rows, cols)