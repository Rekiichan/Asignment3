import cv2 as cv
import numpy as np

def HighBoostFiltering(image,Scale_Factor):
    hsv_img = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    h,s,gray_img = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]
    resultant_image = gray_img.copy()
    for i in range(1,gray_img.shape[0]-1):
        for j in range(1,gray_img.shape[1]-1):
            blur_factor = gray_img[i-1, j-1] + gray_img[i-1, j] - gray_img[i-1, j+1] + gray_img[i, j-1] + gray_img[i, j] + gray_img[i, j+1] + gray_img[i+1, j+1] + gray_img[i+1, j] + gray_img[i+1, j+1]
            blur_factor /= 9
            mask = Scale_Factor*gray_img[i, j] - blur_factor
            resultant_image[i, j] = gray_img[i, j] + mask

    hsv_img = cv.merge([h,s,gray_img])
    BGR_Image = cv.cvtColor(hsv_img,cv.COLOR_HSV2BGR)
    return resultant_image, BGR_Image

if __name__ == "__main__":
    Image = cv.imread("4_2.bmp")
    Gray_Image, Filtered_Image = HighBoostFiltering(Image,3)
    cv.imshow("Original Image", Image)
    cv.imshow("Filtered Gray Image", Gray_Image)
    cv.imshow("Filtered Image", Filtered_Image)
    cv.waitKey()
    cv.destroyAllWindows()