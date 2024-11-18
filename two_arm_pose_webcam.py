# Credit to "https://github.com/lucasdevit0/Movement-Tracking"


import mediapipe as mp    #MediaPipe Traking - Machine Learning Pipeline
import numpy as np        #Operations
import cv2                #Open CV

#Initialize utilities and packages
mp_drawing = mp.solutions.drawing_utils  #drawing utilities
mp_holistic = mp.solutions.holistic      #Importing holistic model
mp_pose = mp.solutions.pose

#Calculate angle between three joints
def calculate_angles(left_shoulder, right_shoulder, left_elbow, left_wrist, right_elbow, right_wrist):
    '''
    Args:
        left_shoulder: [left shoulder x coord, left shoulder y coord]
        right_shoulder: [right shoulder x coord, right shoulder y coord]
        elbow: [elbow x coord, elbow y coord]
        wrist: [wrist x coord, wrist y coord]
    '''
    left_s = np.array(left_shoulder)
    l_elb = np.array(left_elbow)
    l_wri = np.array(left_wrist)
    right_s = np.array(right_shoulder)
    r_elb = np.array(right_elbow)
    r_wri = np.array(right_wrist)
    

    # This math uses the formula: result = atan2(P3.y - P1.y, P3.x - P1.x) -
    # atan2(P2.y - P1.y, P2.x - P1.x), where P1 is the central angle
    
    radians = np.arctan2(l_wri[1]-l_elb[1],l_wri[0]-l_elb[0]) - np.arctan2(left_s[1]-l_elb[1],left_s[0]-l_elb[0])
    #convert to absolute value degrees
    l_elbow_angle = np.abs((radians*180.0)/(np.pi))
    
    if l_elbow_angle > 180.0:
        l_elbow_angle = 360-l_elbow_angle

    # In practice, the elbow angle will only be between ~20 and 180 degrees

    
    # Calculate shoulder angle --> finds the angle formed between your left shoulder,
    # right shoulder, and elbow, then subtracts 90 degrees to get angle between the
    # arm and torso.

    # radians = np.arctan2(elb[1]-left_s[1],elb[0]-left_s[0]) - np.arctan2(right_s[1]-left_s[1],right_s[0]-left_s[0])
    radians = np.arctan2(l_elb[1]-left_s[1],l_elb[0]-left_s[0]) - np.arctan2(0,right_s[0]-left_s[0])
    #convert to absolute value degrees
    l_shoulder_angle = np.abs((radians*180.0)/(np.pi)) - 90.0
    
    if l_shoulder_angle < 0:
        l_shoulder_angle = 0
    if l_shoulder_angle > 180:
        l_shoulder_angle = 180


    # The shoulder angle will range between 0 and ~160 degrees, with 0 being arm at side
    # and 160 being the arm is straight up.

    # Find right elbow angle
    radians = np.arctan2(r_wri[1]-r_elb[1],r_wri[0]-r_elb[0]) - np.arctan2(right_s[1]-r_elb[1],right_s[0]-r_elb[0])
    #convert to absolute value degrees
    r_elbow_angle = np.abs((radians*180.0)/(np.pi))
    
    if r_elbow_angle > 180.0:
        r_elbow_angle = 360-r_elbow_angle


    # Find right shoulder angle NOT WORKING
    radians = np.arctan2(r_elb[1]-right_s[1],r_elb[0]-right_s[0]) - np.arctan2(0,left_s[0]-right_s[0])
    #convert to absolute value degrees
    r_shoulder_angle = np.abs((radians*180.0)/(np.pi)) - 90.0
    
    if r_shoulder_angle < 0:
        r_shoulder_angle = 0
    if r_shoulder_angle > 180:
        r_shoulder_angle = 180
    
        
    return l_elbow_angle, l_shoulder_angle, r_elbow_angle, r_shoulder_angle



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
        #print(results.pose_landmarks)
        #face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        #Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

        
        #Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            #Get Coordinates
            left_shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]
            right_elbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]
            #Calculate angle
            l_elbow_angle, l_shoulder_angle, r_elbow_angle, r_shoulder_angle = calculate_angles(left_shoulder, right_shoulder, left_elbow, left_wrist, right_elbow, right_wrist)
            print(l_shoulder_angle)
            #Visualize angles
            cv2.putText(image,str(l_elbow_angle),tuple(np.multiply(left_elbow,[640,480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image,str(l_shoulder_angle),tuple(np.multiply(left_shoulder,[640,480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image,str(r_shoulder_angle),tuple(np.multiply(right_shoulder,[640,480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image,str(r_elbow_angle),tuple(np.multiply(right_elbow,[640,480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            # print(landmarks)
        except:
            pass


        # Draw pose detection
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
