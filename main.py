import cv2 as cv

originalImage = cv.imread("friends.jpg")
grayImage = cv.cvtColor(originalImage, cv.COLOR_BGR2GRAY)
faceCascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")
detectedFaces = faceCascade.detectMultiScale(grayImage, 1.1)

for (col, row, width, height) in detectedFaces:
    cv.rectangle(
        originalImage,
        (col, row),
        (col + width, row + height),
        (0, 0, 255),
        5
    )

#smallImage = cv.resize(originalImage, (0,0), fx=0.5, fy=0.5)
cv.imshow('Friends', originalImage)
cv.waitKey(0)
cv.destroyAllWindows()