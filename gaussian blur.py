import cv2

if __name__ == "__main__":
    image = cv2.imread('test_descreen.tif')
    blurred_image = cv2.GaussianBlur(image, (0,0), sigmaX=0.8, sigmaY=0.8)
    cv2.imwrite('blurred_image.tif', blurred_image)

    