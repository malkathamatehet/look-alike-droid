# Credit to "https://github.com/lucasdevit0/Movement-Tracking"


import mediapipe as mp    #MediaPipe Traking - Machine Learning Pipeline
import numpy as np        #Operations
import cv2                #Open CV

#Initialize utilities and packages
mp_drawing = mp.solutions.drawing_utils  #drawing utilities
mp_holistic = mp.solutions.holistic      #Importing holistic model
mp_pose = mp.solutions.pose


def calculate_angles(left_shoulder, right_shoulder, left_elbow, left_wrist, right_elbow, right_wrist):
    '''
    Our two degrees of freedom angle calculation function, which outputs all 
    necessary angles to control both arms of the LAD in 2 DOF.

    Args:
        left_shoulder: [left shoulder x coord, left shoulder y coord, left shoulder z coord]
        right_shoulder: [right shoulder x coord, right shoulder y coord, right shoulder z coord]
        left_elbow: [left elbow x coord, left elbow y coord, left elbow z coord]
        left_wrist: [left wrist x coord, left wrist y coord, left wrist z coord]
        right_elbow: [right elbow x coord, right elbow y coord, right elbow z coord]
        right_wrist: [right wrist x coord, right wrist y coord, right wrist z coord]

    Returns:
        angles: A dict containing the following angles:
            "ls_xy": (int) The xy angle of the left shoulder.
            "ls_xz": (int) The xz angle of the left shoulder.
            "ls_yz": (int) The yz angle of the left shoulder, aka the "twisting" angle.
            "l_elbow": (int) The acute angle of the left elbow.
            "rs_xy": (int) The xy angle of the right shoulder.
            "rs_xz": (int) The xz angle of the right shoulder.
            "rs_yz": (int) The yz angle of the right shoulder, aka the "twisting" angle.
            "r_elbow": (int) The acute angle of the right elbow.
    '''

    ls_pos = np.array(left_shoulder)
    l_elb_pos = np.array(left_elbow)
    l_wri_pos = np.array(left_wrist)
    rs_pos = np.array(right_shoulder)
    r_elb_pos = np.array(right_elbow)
    r_wri_pos = np.array(right_wrist)


    angles = {}


    # Calculate left shoulder xy angle
    radians = np.arctan2(l_elb_pos[1]-ls_pos[1],l_elb_pos[0]-ls_pos[0]) - np.arctan2(0,rs_pos[0]-ls_pos[0])
    #convert to absolute value degrees
    angles["ls_xy"] = np.abs((radians*180.0)/(np.pi)) - 90.0

    # Calculate left shoulder xz angle
    radians = np.arctan2(l_elb_pos[2]-ls_pos[2],l_elb_pos[0]-ls_pos[0])
    #convert to degrees and modify to have normal movement range be 0 to 180
    angles["ls_xz"] = -1 * (radians*180.0)/(np.pi) + 90

    # Calculate left shoulder yz angle
    radians = np.arctan2((l_wri_pos[1]-l_elb_pos[1]), -1 * (l_wri_pos[2]-l_elb_pos[2]))
    #convert to degrees and to be from 0 to 180
    angles["ls_yz"] = -1 * (radians*180.0)/(np.pi) + 90

    # Calculate left elbow angle
    radians = np.arctan2(l_wri_pos[1]-l_elb_pos[1],l_wri_pos[0]-l_elb_pos[0]) - np.arctan2(ls_pos[1]-l_elb_pos[1],ls_pos[0]-l_elb_pos[0])
    #convert to absolute value degrees
    l_elbow_angle = np.abs((radians*180.0)/(np.pi))
    
    if l_elbow_angle > 180.0:
        l_elbow_angle = 360-l_elbow_angle

    angles["l_elbow"] = l_elbow_angle

    # Calculate right shoulder xy angle
    radians = np.arctan2(r_elb_pos[1]-rs_pos[1], -1 * (r_elb_pos[0]-rs_pos[0])) - np.arctan2(0,ls_pos[0]-rs_pos[0])
    #convert to absolute value degrees
    angles["rs_xy"] = -1 * (radians*180.0)/(np.pi) + 90.0

    # Calculate right shoulder xz angle
    radians = np.arctan2(r_elb_pos[2]-rs_pos[2], -1 * (r_elb_pos[0]-rs_pos[0]))
    #convert to degrees and modify to have normal movement range be 0 to 180
    angles["rs_xz"] = -1 * (radians*180.0)/(np.pi) + 90

    # Calculate right shoulder yz angle
    radians = np.arctan2((r_wri_pos[1]-r_elb_pos[1]), -1 * (r_wri_pos[2]-r_elb_pos[2]))
    #convert to degrees and to be from 0 to 180
    angles["rs_yz"] = -1 * (radians*180.0)/(np.pi) + 90

    # Calculate right elbow angle
    radians = np.arctan2(r_wri_pos[1]-r_elb_pos[1],r_wri_pos[0]-r_elb_pos[0]) - np.arctan2(rs_pos[1]-r_elb_pos[1],rs_pos[0]-r_elb_pos[0])
    #convert to absolute value degrees
    r_elbow_angle = np.abs((radians*180.0)/(np.pi))
    
    if r_elbow_angle > 180.0:
        r_elbow_angle = 360-r_elbow_angle

    angles["r_elbow"] = r_elbow_angle


    # Make sure all angles fall with [0, 180], the range of our motors
    for key, angle in angles.items():
        if angle < 0:
            angles[key] = 0
        if angle > 180:
            angles[key] = 180


    return angles



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
            left_elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y,
                     landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].z]
            left_wrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y,
                     landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].z]
            right_elbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y,
                     landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].z]
            right_wrist = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y,
                     landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].z]

            

            angles = calculate_angles(left_shoulder, right_shoulder, left_elbow, left_wrist, right_elbow, right_wrist)
            
            
            ## Display angles

            # Display left shoulder xy angle
            # cv2.putText(image,str(round(angles["ls_xy"])),tuple(np.multiply(left_shoulder[0:2],[640,480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Display left shoulder xz angle
            # cv2.putText(image,str(round(angles["ls_xz"])),tuple(np.multiply(left_shoulder[0:2],[640,480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Display left shoulder yz angle
            # cv2.putText(image,str(round(angles["ls_yz"])),tuple(np.multiply(left_shoulder[0:2],[640,480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Display left elbow angle
            # cv2.putText(image,str(round(angles["l_elbow"])),tuple(np.multiply(left_elbow[0:2],[640,480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            # Display right shoulder xy angle
            # cv2.putText(image,str(round(angles["rs_xy"])),tuple(np.multiply(right_shoulder[0:2],[640,480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            # Display right shoulder xz angle
            # cv2.putText(image,str(round(angles["rs_xz"])),tuple(np.multiply(right_shoulder[0:2],[640,480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            # Display right shoulder yz angle
            # cv2.putText(image,str(round(angles["rs_yz"])),tuple(np.multiply(right_shoulder[0:2],[640,480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            # Display right elbow angle
            # cv2.putText(image,str(round(angles["r_elbow"])),tuple(np.multiply(right_elbow[0:2],[640,480]).astype(int)),
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
