import cv2

img = cv2.imread('assets/picofme.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (400, 400))
# img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()