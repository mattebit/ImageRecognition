import cv2

imagePath = "hand3.jpg"
cascPath = "haarPalm.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hands = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.06,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print( "Found {0} hands!".format(len(hands)))

# Draw a rectangle around the faces
for (x, y, w, h) in hands:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("hands found" ,image)
cv2.waitKey(0)
