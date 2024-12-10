# The brain of our LAD! This controls all the motors, LEDs, and vision capabilities of LAD
import time

import mediapipe as mp    # MediaPipe Tracking - Machine Learning Pipeline
import numpy as np        # Operations
import cv2                # Open CV

# Initialize utilities and packages
mp_drawing = mp.solutions.drawing_utils  # drawing utilities
mp_pose = mp.solutions.pose # pose detection

# Camera Setup
from picamera2 import Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": 'RGB888', "size": (640, 480)}))
picam2.start()

# Driver Setup
import board

# LED Setup
import neopixel
pos_leds = neopixel.NeoPixel(board.D18, 30) 

# Servo Hat Setup
import busio
import adafruit_pca9685
i2c = busio.I2C(board.SCL, board.SDA)
hat = adafruit_pca9685.PCA9685(i2c)

from adafruit_servokit import ServoKit
kit = ServoKit(channels=16)

# Define motors
ls_xy_motor = kit.servo[0]
ls_xz_motor = kit.servo[0]
ls_yz_motor = kit.servo[0]
le_motor = kit.servo[0]

rs_xy_motor = kit.servo[0]
rs_xz_motor = kit.servo[0]
rs_yz_motor = kit.servo[0]
re_motor = kit.servo[0]

# Set all angles to 0 - resting/reset position
ls_xy_motor.angle = 0
ls_xz_motor.angle = 0
ls_yz_motor.angle = 0
le_motor.angle = 0

rs_xy_motor.angle = 0
# rs_xz_motor.angle = 0
rs_yz_motor.angle = 180
re_motor.angle = 0

def calculate_angles(landmarks):
    """
    Function to calculate the angles needed to mimic a person's pose with two arms in 2DoF

    Args:
    A dictionary containing:
        left_shoulder: list containing the x, y, and z coordinates of the left shoulder landmark
        right_shoulder: list containing the x, y, and z coordinates of the right shoulder landmark
        left_elbow: list containing the x, y, and z coordinates of the left elbow landmark
        left_wrist: list containing the x, y, and z coordinates of the left wrist landmark
        right_elbow: list containing the x, y, and z coordinates of the right elbow landmark
        right_wrist: list containing the x, y, and z coordinates of the right wrist landmark

    Returns: 
    A dictionary that contains:
        ls_xy: The xy angle of the left shoulder in relation to the torso
        ls_xz: The xz angle of the left shoulder in relation to the torso
        ls_yz: The yz angle of the left shoulder in relation to the torso
        le: The angle of the left elbow in relation to the left bicep
        rs_xy: The xy angle of the right shoulder in relation to the torso
        rs_xz: The xz angle of the right shoulder in relation to the torso
        rs_yz: The yz angle of the right shoulder in relation to the torso
        re: The angle of the right elbow in relation to the right bicep

    """
    # Define lists as numpy arrays for easier calculations
    ls_pos = np.array(landmarks[0])
    l_elb_pos = np.array(landmarks[2])
    l_wri_pos = np.array(landmarks[4])
    rs_pos = np.array(landmarks[1])
    r_elb_pos = np.array(landmarks[3])
    r_wri_pos = np.array(landmarks[5])

    # Declare dictionary to return
    angles = {}

    # Calculate left shoulder xy angle
    ls_xy_radians = np.arctan2(l_elb_pos[1]-ls_pos[1],l_elb_pos[0]-ls_pos[0]) - np.arctan2(0,rs_pos[0]-ls_pos[0])
    # Convert to degrees and modify to have normal movement range be 0 to 180
    angles["ls_xy"] = np.abs((ls_xy_radians*180.0)/(np.pi)) - 90.0

    # Calculate left shoulder xz angle
    ls_xz_radians = np.arctan2(l_elb_pos[2]-ls_pos[2],l_elb_pos[0]-ls_pos[0])
    # Convert to degrees and modify to have normal movement range be 0 to 180
    angles["ls_xz"] = -1 * (ls_xz_radians*180.0)/(np.pi)

    if angles["ls_xz"] < 0:
        angles["ls_xz"] = 0

    # Calculate left shoulder yz angle
    ls_yz_radians = np.arctan2((l_wri_pos[1]-l_elb_pos[1]), -1 * (l_wri_pos[2]-l_elb_pos[2]))
    # Convert to degrees and modify to have normal movement range be 0 to 180
    angles["ls_yz"] = -1 * (ls_yz_radians*180.0)/(np.pi) + 90

    # Calculate left elbow angle
    le_radians = np.arctan2(l_wri_pos[1]-l_elb_pos[1],l_wri_pos[0]-l_elb_pos[0]) - np.arctan2(ls_pos[1]-l_elb_pos[1],ls_pos[0]-l_elb_pos[0])
    # Convert to degrees
    angles["l_elbow"] = 180 - np.abs((le_radians*180.0)/(np.pi))

    # Calculate right shoulder xy angle
    rs_xy_radians = np.arctan2(r_elb_pos[1]-rs_pos[1], -1 * (r_elb_pos[0]-rs_pos[0])) - np.arctan2(0,ls_pos[0]-rs_pos[0])
    # Convert to degrees and modify to have normal movement range be 0 to 180
    angles["rs_xy"] = -1 * (rs_xy_radians*180.0)/(np.pi) + 90.0

    # Calculate right shoulder xz angle
    rs_xz_radians = np.arctan2(r_elb_pos[2]-rs_pos[2], -1 * (r_elb_pos[0]-rs_pos[0]))
    # Convert to degrees and modify to have normal movement range be 0 to 180
    angles["rs_xz"] = -1 * (rs_xz_radians*180.0)/(np.pi) + 90

    # Calculate right shoulder yz angle
    rs_yz_radians = np.arctan2((r_wri_pos[1]-r_elb_pos[1]), -1 * (r_wri_pos[2]-r_elb_pos[2]))
    # Convert to degrees and modify to have normal movement range be 0 to 180
    angles["rs_yz"] = 180 - (-1 * (rs_yz_radians*180.0)/(np.pi) + 90)

    # Calculate right elbow angle
    radians = np.arctan2(r_wri_pos[1]-r_elb_pos[1],r_wri_pos[0]-r_elb_pos[0]) - np.arctan2(rs_pos[1]-r_elb_pos[1],rs_pos[0]-r_elb_pos[0])
    # Convert to degrees
    angles["r_elbow"] = 180 - np.abs((radians*180.0)/(np.pi))

    # Make sure all angles fall with [0, 180], the range of our motors
    for key, angle in angles.items():
        if angle < 0:
            angles[key] = 0
        if angle > 180:
            angles[key] = 180


    return angles

# Initiate Pose model
with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    
    frame_count = -1

    # Open camera
    while True:

        # Read camera feed
        frame = picam2.capture_array()
        
        frame_count += 1

        # Only run pose detection model on every 5 frames (to reduce load for raspi)
        if frame_count % 5 == 0:

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            
            # Make Detections
            results = pose.process(image)              
            
            # Recolor image back to BGR for rendering
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

            # Declare user position boolean for LED usage
            user_in_position = False

             # Check if relevant landmarks are detected 
            if results.pose_landmarks:
                
                # Select only relevant landmarks
                landmarks = results.pose_landmarks.landmark[11:17]

                # Check that all landmarks have a visibility over 0.9
                user_in_position = all(
                    landmark.visibility > 0.9
                    for landmark in landmarks
                )

                # User is in position 
                if user_in_position:
                    #leds green

                    # Get landmark values
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
                    
                    landmark_list = [left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist]

                    # Calculate the angles
                    angles = calculate_angles(landmark_list)

                    # Move motors to correct angles
                    ls_xy_motor.angle = angles["ls_xy"]
                    ls_xz_motor.angle = angles["ls_xz"]
                    ls_yz_motor.angle = angles["ls_yz"]
                    le_motor.angle = angles["l_elbow"]

                    rs_xy_motor.angle = angles["rs_xy"]
                    rs_xz_motor.angle = angles["rs_xz"]
                    rs_yz_motor.angle = angles["rs_yz"]
                    re_motor.angle = angles["r_elbow"]

                else:
                    print("nope")
                    #leds red

            #Hit "q" to kill camera
            if cv2.waitKey(10) & 0xFF == ord('q'):         
                break
picam2.close()
cv2.destroyAllWindows()









