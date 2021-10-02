import cv2 as cv
from os import path

face_classifier = cv.CascadeClassifier(
    path.abspath('resources/myfacedetector.xml'))

eye_classifier = cv.CascadeClassifier(
    path.abspath('resources/haarcascade_eye.xml'))

mounth_classifier = cv.CascadeClassifier(
    path.abspath('resources/mounth.xml'))


img = cv.imread(path.abspath('assets/img/005.png'))

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(
    gray_img,
    scaleFactor=1.1,
    minNeighbors=5,
    flags=cv.CASCADE_SCALE_IMAGE
)

for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Getting face coordenates
    gray_face = gray_img[y:y+h, x:x+w]
    face = img[y:y+h, x:x+w]

    # Detecting Eyes
    eyes = eye_classifier.detectMultiScale(
        gray_face,
        scaleFactor=1.00003,
        minNeighbors=1,
        flags=cv.CASCADE_SCALE_IMAGE
    )

    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(face, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

    # Detecting Mounths
    mounth = mounth_classifier.detectMultiScale(
        gray_face,
        scaleFactor=1.007,
        minNeighbors=20,
        flags=cv.CASCADE_SCALE_IMAGE,
        maxSize=(40,25),
    )

    for (ex, ey, ew, eh) in mounth:
        cv.rectangle(face, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

cv.imshow("Result", img)
cv.waitKey(0)
