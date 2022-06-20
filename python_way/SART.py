from skimage.transform import iradon_sart
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import cv2
from skimage.transform import radon, rescale

image = skimage.io.imread(fname="D:/College Works/Spring 2022/Advanced Computer Vision/projections/16Projection95.tif")


theta = np.linspace(0., 180., max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta)

reconstruction_sart = iradon_sart(sinogram, theta=theta, dtype = np.float64)


reconstruction_sart2 = iradon_sart(sinogram, theta=theta,
                                   image=reconstruction_sart)

# cv2.imshow("SART",reconstruction_sart2)
# cv2.waitKey(0)
sharpen_filter = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
sharped_img = cv2.filter2D(reconstruction_sart2, -1, sharpen_filter)
# cv2.imshow('Reconstruction SART 1',reconstruction_sart)
# cv2.imshow('Reconstruction SART 2',reconstruction_sart2)
cv2.imshow('Reconstruction SART Sharpened',sharped_img)
cv2.waitKey(0)
# # Normalizing the data for saving the image for SART with 1 iteration
# reconstruction_sart = reconstruction_sart - reconstruction_sart.min()
# reconstruction_sart = reconstruction_sart / reconstruction_sart.max() * 255
# reconstructionone = np.uint8(reconstruction_sart)
# # Normalizing the data for saving the image for SART with 2 iteration
# reconstruction_sart2 = reconstruction_sart2 - reconstruction_sart2.min()
# reconstruction_sart2 = reconstruction_sart2 / reconstruction_sart2.max() * 255
# reconstructiontwo = np.uint8(reconstruction_sart2)
#
# cv2.imwrite("D:/College Works/Spring 2022/Advanced Computer Vision/Reconstruction/SARTRecon1.png",reconstructionone)
# cv2.imwrite("D:/College Works/Spring 2022/Advanced Computer Vision/Reconstruction/SARTRecon2.png",reconstructiontwo)
