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

faces = face_classifier.detectMultiScale(gray_img, 1.05, 9)

for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Getting face coordenates
    gray_face = gray_img[y:y+h, x:x+w]
    face = img[y:y+h, x:x+w]

    # Detecting Eyes
    eyes = eye_classifier.detectMultiScale(gray_face, 1.0001, 2)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(face, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

    # Detecting Mounths
    mounth = mounth_classifier.detectMultiScale(gray_face, 1.075, 4)
    for (ex, ey, ew, eh) in mounth:
        cv.rectangle(face, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)





cv.imshow("Result", img)
cv.waitKey(0)
