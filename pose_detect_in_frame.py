# Credit to "https://github.com/lucasdevit0/Movement-Tracking"

import time

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

# Initalize indicator of whether the user is in position
user_in_position = False

last_print_time = time.time()

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


        # Check if relevant landmarks are detected for LED display
        if results.pose_landmarks:

            current_time = time.time()
                
            # Print landmark visibilities every four seconds
            if current_time - last_print_time >= 4:
                last_print_time = current_time

            
                # Extract relevant landmarks
                # left_shoulder = results.pose_landmarks.landmark[11]
                # right_shoulder = results.pose_landmarks.landmark[12]
                # left_elbow = results.pose_landmarks.landmark[13]
                # right_elbow = results.pose_landmarks.landmark[14]
                # left_wrist = results.pose_landmarks.landmark[15]
                # right_wrist = results.pose_landmarks.landmark[16]
                
                # Print their visibility scores
                # print(f"Left Shoulder Visibility: {left_shoulder.visibility}")
                # print(f"Right Shoulder Visibility: {right_shoulder.visibility}")
                # print(f"Left Elbow Visibility: {left_elbow.visibility}")
                # print(f"Right Elbow Visibility: {right_elbow.visibility}")
                # print(f"Left Wrist Visibility: {left_wrist.visibility}")
                # print(f"Right Wrist Visibility: {right_wrist.visibility}")
                # print("\n")

                
            # Select landmarks 11-16, which are the left and right shoulders, elbows, 
            # and wrists
            landmarks = results.pose_landmarks.landmark[11:17]

            # Check that all landmarks have a visibility over 0.9
            user_in_position = all(
                landmark.visibility > 0.9
                for landmark in landmarks
            )

            # Print the result
            print("User is in position!" if user_in_position else "User is not in position.")



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
