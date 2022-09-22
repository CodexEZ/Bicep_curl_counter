from audioop import avg
import cv2
import mediapipe as mp
import numpy as np
import requests


#calculates the amount of calorie burnt in total
def caloriemeter(reps, weight):
    calorie_per_rep = weight * 0.00267
    total_calorie = calorie_per_rep * reps
    return total_calorie

#calculates the angle between shoulder ,elbow and wrist 
def calculate_angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle>180:
        angle = 360-angle
    return angle

mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose

cap = cv2.VideoCapture(0)

counter=0
Rcounter = 0
stage=None
Rstage=None
weight = int(input("Enter the weight you're lifting"))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)as pose:
    #starting video capture
    while cap.isOpened():
        ret,frame = cap.read()

        #image recolor to rgb
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False

        #Make detection
        results = pose.process(image)

        #recolor back to bgr
        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        #Extracting Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            #right side
            Rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            Relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            Rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            angle = calculate_angle(shoulder,elbow,wrist)
            Rangle = calculate_angle(Rshoulder,Relbow,Rwrist)

            #visualize image
            cv2.putText(image, str(angle), tuple(np.multiply(elbow,[640,480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2,cv2.LINE_AA)
            cv2.putText(image, str(Rangle), tuple(np.multiply(Relbow,[640,480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2,cv2.LINE_AA)

            if angle>160:
                stage = "down"
            if angle < 30 and stage=="down":
                stage = "up"
                counter +=1
                print(counter)
            if Rangle>160:
                Rstage="down"
            if Rangle<30 and Rstage=="down":
                Rstage="up"
                Rcounter+=1
                
        except:
            pass
        #setup status box        
        
        cv2.rectangle(image,(0,0),(250,73),(245,117,16),-1)
        cv2.rectangle(image,(0,84),(250,166),(245,117,16),-1)

        #rep data

        cv2.putText(image,'REPS', (15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,str(counter), (10,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(image,'STAGE', (65,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,"LEFT-SIDE",(120,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,cv2.LINE_AA)
        cv2.putText(image,stage,(80,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)

        #rep data for right hand
        cv2.putText(image,"REPS",(15,96),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,str(Rcounter),(10,144),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(image,"STAGE",(65,96),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,"RIGHT-SIDE",(120,96),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,cv2.LINE_AA)
        cv2.putText(image,Rstage,(80,144),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)

        #render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
        cv2.imshow("Mediapipe feed",image)
        if cv2.waitKey(10) & 0xFF==ord('q'):
            #calculating number of curls in total
            total_curls = counter+Rcounter
            #posting data to ThingsSpeak using API of the channel
            url = f"https://api.thingspeak.com/update?api_key=2T20HA5UPJNNPJ86&field1={counter}&field2={Rcounter}&field3={caloriemeter(total_curls,weight)}"
            response = requests.request("GET",url)
            break
cap.release()
cv2.destroyAllWindows()
