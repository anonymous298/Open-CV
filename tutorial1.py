import cv2

image = cv2.imread('assets/humans.jpg')
# print(image.shape)


resized = cv2.resize(image, (400, 400))

# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
# print(cv2.COLOR_BGR2GRAY)

# flipped = cv2.flip(gray, -1)

# edges = cv2.Canny(resized, 100, 200)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(resized, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('image', resized)
cv2.waitKey()