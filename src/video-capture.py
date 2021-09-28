import cv2 as cv
from os import path

face_classifier = cv.CascadeClassifier(
    path.abspath('resources/myfacedetector.xml'))

eye_classifier = cv.CascadeClassifier(
    path.abspath('resources/haarcascade_eye.xml'))

mounth_classifier = cv.CascadeClassifier(
    path.abspath('resources/mounth.xml'))


cap = cv.VideoCapture(0)

while True:
    _, img = cap.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Getting face coordenates
        gray_face = gray[y:y+h, x:x+w]
        face = img[y:y+h, x:x+w]

        # Detecting Eyes
        eyes = eye_classifier.detectMultiScale(gray_face, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(face, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

        # Detecting Mounths
        mounth = mounth_classifier.detectMultiScale(gray_face, 1.3, 5)
        for (ex, ey, ew, eh) in mounth:
            cv.rectangle(face, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv.imshow('Video', img)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()
