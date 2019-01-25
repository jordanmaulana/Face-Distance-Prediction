#!/usr/bin/python

import cv2
cap = cv2.VideoCapture(0)
ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
ret = cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

f = open('dataset.csv','a')

actual_distance = 30 #cm

def drawBoxAndWriteText(findfaces):
    for (x, y, w, h) in findfaces:
        line = "%d,"% (h) +str(actual_distance)+"\n" 
        f.write(line)
        cv2.rectangle(color, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(color, str(h), (x, y+h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

# Main Program
while(True):
    # Capture frame-by-frame
    ret, color = cap.read()
    # Detect face
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
    	gray, 
    	scaleFactor=1.1,
    	minNeighbors=5,
    	minSize=(30,30)
    )
    # Draw box and write text
    drawBoxAndWriteText(faces)
    # Show the resulting frame
    cv2.imshow('color', color)
    # check if key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        f.close()
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


