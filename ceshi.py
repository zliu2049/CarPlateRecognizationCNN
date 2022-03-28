import os

import cv2
import numpy as np


def gaussian_blur(image, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def canny(image, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(image, low_threshold, high_threshold)


def pre_process(orig_img):
    cv2.namedWindow("orig_img", 0)
    cv2.resizeWindow("orig_img", 640, 480)
    cv2.imshow('orig_img', orig_img)

    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)  # convert to grayscale image
    cv2.namedWindow("gray_img", 0)
    cv2.resizeWindow("gray_img", 640, 480)
    cv2.imshow('gray_img', gray_img)

    kernel_size = 5
    gauss_gray = gaussian_blur(gray_img, kernel_size)  # gaussian blue
    cv2.namedWindow("blur", 0)
    cv2.resizeWindow("blur", 640, 480)
    cv2.imshow('blur', gauss_gray)

    low_threshold = 50
    high_threshold = 150
    cv2.namedWindow("canny", 0)
    cv2.resizeWindow("canny", 640, 480)
    canny_edges = canny(gauss_gray, low_threshold, high_threshold)  # canny edge finding
    cv2.imshow('canny', canny_edges)

    '''sobel_img = cv2.Sobel(gauss_gray, cv2.CV_16S, 1, 0, ksize=3)
    sobel_img = cv2.convertScaleAbs(sobel_img)  # sobel edge finding
    cv2.imshow('sobel', sobel_img)'''

    hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)  # convert to HSV image
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    cv2.namedWindow("hsv", 0)
    cv2.resizeWindow("hsv", 640, 480)
    cv2.imshow('hsv', hsv_img)

    blue_img = (((h > 11) & (h < 34)) | ((h > 100) & (h < 124)) & (s > 70) & (v > 70))
    blue_img = blue_img.astype('float32')  # find blue area [100, 124]
    cv2.namedWindow("blue", 0)
    cv2.resizeWindow("blue", 640, 480)
    cv2.imshow('blue', blue_img)

    mix_img = np.multiply(canny_edges, blue_img)  # find blue edge
    cv2.namedWindow("mix", 0)
    cv2.resizeWindow("mix", 640, 480)
    cv2.imshow('mix', mix_img)

    mix_img = mix_img.astype(np.uint8)

    ret, binary_img = cv2.threshold(mix_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # binarization
    cv2.namedWindow("binary", 0)
    cv2.resizeWindow("binary", 640, 480)
    cv2.imshow('binary', binary_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)  # morphology operation
    cv2.namedWindow("close", 0)
    cv2.resizeWindow("close", 640, 480)
    cv2.imshow('close', close_img)
    cv2.waitKey(0)
    return close_img


if __name__ == '__main__':
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    car_plate_w, car_plate_h = 136, 36
    char_w, char_h = 20, 20
    plate_model_path = './carIdentityData/model/plate_recongnize/model.ckpt-510.meta'
    char_model_path = './carIdentityData/model/char_recongnize/model.ckpt-600.meta'
    img = cv2.imread('../images/pictures/1.jpg')

    pred_img = pre_process(img)  # preprocessing
