# Credit to "https://github.com/lucasdevit0/Movement-Tracking"


import mediapipe as mp    #MediaPipe Traking - Machine Learning Pipeline
import numpy as np        #Operations
import cv2                #Open CV

#Initialize utilities and packages
mp_drawing = mp.solutions.drawing_utils  #drawing utilities
mp_holistic = mp.solutions.holistic      #Importing holistic model
mp_pose = mp.solutions.pose

#Calculate angle between three joints
def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    #convert to absolute value degrees
    angle = np.abs((radians*180.0)/(np.pi))
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle



#Default Hardware Capture device
cap = cv2.VideoCapture(0)                             

#Initiate Holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    #Open Camera
    while cap.isOpened(): 
        #Read Camera Feed
        ret, frame = cap.read()    
        
        #Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        
        #Make Detections
        results = holistic.process(image)              
        #print(results.face_landmarks)
        #face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        #Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

        
        #Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            #Get Coordinates
            shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]
            #Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            #Visualzie angle
            cv2.putText(image,str(angle),tuple(np.multiply(elbow,[640,480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            print(landmarks)
        except:
            pass

        #1. Draw Face Landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color = (255,255,0),thickness = 1, circle_radius = 1),     #dot color
                                  mp_drawing.DrawingSpec(color = (192,192,192),thickness = 1, circle_radius = 1))   #connection color
        
        #2. Draw right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color = (80,22,10),thickness = 2, circle_radius = 4),
                                  mp_drawing.DrawingSpec(color = (80,44,121),thickness = 2, circle_radius = 2))

        #3. Draw left hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color = (121,22,76),thickness = 2, circle_radius = 4),
                                  mp_drawing.DrawingSpec(color = (121,44,250),thickness = 2, circle_radius = 2))
        #4. Draw pose detection
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color = (245,117,66),thickness = 2, circle_radius = 4),
                                  mp_drawing.DrawingSpec(color = (245,66,230),thickness = 2, circle_radius = 2))
        
        
        #Render Video
        cv2.imshow('Holistic Model Detection',image)   
        
        #Hit "q" to kill camera
        if cv2.waitKey(10) & 0xFF == ord('q'):         
            break
    
cap.release()
cv2.destroyAllWindows()
