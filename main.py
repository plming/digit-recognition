import cv2


# load lenna image
img = cv2.imread('lenna.png')

# display image
cv2.imshow('Lenna', img)
cv2.waitKey()
cv2.destroyAllWindows()