# Credit to "https://github.com/lucasdevit0/Movement-Tracking"


import mediapipe as mp    #MediaPipe Traking - Machine Learning Pipeline
import numpy as np        #Operations
import cv2                #Open CV

#Initialize utilities and packages
mp_drawing = mp.solutions.drawing_utils  #drawing utilities
mp_holistic = mp.solutions.holistic      #Importing holistic model
mp_pose = mp.solutions.pose


def calculate_angles(left_shoulder, right_shoulder, elbow, wrist):
    '''
    Our two degrees of freedom angle calculation function, which outputs all 
    necessary angle to control the left arm of the LAD in 2 DOF.

    Currently only outputting left arm angles, right arm to come!

    Args:
        left_shoulder: [left shoulder x coord, left shoulder y coord, left shoulder z coord]
        right_shoulder: [right shoulder x coord, right shoulder y coord, right shoulder z coord]
        elbow: [elbow x coord, elbow y coord, elbow z coord]
        wrist: [wrist x coord, wrist y coord, wrist z coord]

    Returns:
        ls_xy: The xy angle of the left shoulder.
        ls_xz: The xz angle of the left shoulder.
        ls_yz: The yz angle of the left shoulder, aka the "twisting" angle.
        elbow: The acute angle of the elbow.
    '''

    ls_pos = np.array(left_shoulder)
    elb_pos = np.array(elbow)
    wri_pos = np.array(wrist)
    rs_pos = np.array(right_shoulder)


    # Calculate left shoulder xy angle
    radians = np.arctan2(elb_pos[1]-ls_pos[1],elb_pos[0]-ls_pos[0]) - np.arctan2(0,rs_pos[0]-ls_pos[0])
    #convert to absolute value degrees
    ls_xy_angle = np.abs((radians*180.0)/(np.pi)) - 90.0

    # Calculate left shoulder xz angle
    radians = np.arctan2(elb_pos[2]-ls_pos[2],elb_pos[0]-ls_pos[0])
    #convert to degrees and modify to have normal movement range be 0 to 180
    ls_xz_angle = -1 * (radians*180.0)/(np.pi) + 90

    # Calculate left shoulder yz angle
    radians = np.arctan2((wri_pos[1]-elb_pos[1]), -1 * (wri_pos[2]-elb_pos[2]))
    #convert to degrees and to be from 0 to 180
    ls_yz_angle = -1 * (radians*180.0)/(np.pi) + 90

    # Calculate elbow angle
    radians = np.arctan2(wri_pos[1]-elb_pos[1],wri_pos[0]-elb_pos[0]) - np.arctan2(ls_pos[1]-elb_pos[1],ls_pos[0]-elb_pos[0])
    #convert to absolute value degrees
    elbow_angle = np.abs((radians*180.0)/(np.pi))
    
    if elbow_angle > 180.0:
        elbow_angle = 360-elbow_angle


    # Make sure all angles fall with [0, 180], the range of our motors
    for angle in [ls_xy_angle, ls_xz_angle, ls_yz_angle, elbow_angle]:
        if angle < 0:
            angle = 0
        if angle > 180:
            angle = 180
            

    return ls_xy_angle, ls_xz_angle, ls_yz_angle, elbow_angle



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
            left_shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y,
                             landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].z]
            right_shoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y,
                              landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].z]
            elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y,
                     landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].z]
            wrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y,
                     landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].z]
            #Calculate angle
            # elbow_angle, shoulder_angle = calculate_angles(left_shoulder, right_shoulder, elbow, wrist)
            #Visualize xy angles
            # cv2.putText(image,str(elbow_angle),tuple(np.multiply(elbow,[640,480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            # cv2.putText(image,str(shoulder_angle),tuple(np.multiply(left_shoulder,[640,480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            ls_xy_angle, ls_xz_angle, ls_yz_angle, elbow_angle = calculate_angles(left_shoulder, right_shoulder, elbow, wrist)

            ## Display angles

            # Display left shoulder xy angle
            # cv2.putText(image,str(round(ls_xy_angle)),tuple(np.multiply(left_shoulder[0:2],[640,480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Display left shoulder xz angle
            cv2.putText(image,str(round(ls_xz_angle)),tuple(np.multiply(left_shoulder[0:2],[640,480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Display left shoulder yz angle
            # cv2.putText(image,str(round(ls_yz_angle)),tuple(np.multiply(left_shoulder[0:2],[640,480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Display left shoulder xy angle
            # cv2.putText(image,str(round(elbow_angle)),tuple(np.multiply(elbow[0:2],[640,480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

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
