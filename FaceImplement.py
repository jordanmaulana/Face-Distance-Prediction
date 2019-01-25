#!/usr/bin/python
import pandas as pd
import cv2
from sklearn.linear_model import LinearRegression

cap = cv2.VideoCapture(0)
ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
ret = cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

df = pd.read_csv(r"C:\Users\asus\facedetect\dataset.csv")

x = df['pixel'].values
y = df['cm'].values
x = x[:, None]

reg = LinearRegression()
reg.fit(x,y)

def drawBoxAndWriteText(findfaces):
    for (x, y, w, h) in findfaces:
        z = reg.predict([[h]])
        cv2.rectangle(color, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(color, str(z), (x, y+h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

# Main Program
while(True):
    # Capture frame-by-frame
    ret, color = cap.read()
    # Detect face
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Draw box and write text
    drawBoxAndWriteText(faces)
    # Show the resulting frame
    cv2.imshow('color', color)
    # check if key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


