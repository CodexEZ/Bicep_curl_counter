import cv2
cap=cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame = cap.read()
    cv2.imshow("Mediapipe feed",frame)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        #calculating number of curls in total
        break