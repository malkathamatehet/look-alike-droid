from picamera2 import Picamera2
import cv2
import time
import numpy as np
import board
import busio
import adafruit_pca9685
i2c = busio.I2C(board.SCL, board.SDA)
hat = adafruit_pca9685.PCA9685(i2c)
from adafruit_servokit import ServoKit
kit = ServoKit(channels=16)

shoulder = kit.servo[0]
elbow = kit.servo[1]

shoulder.actuation_range = 45
elbow.actuation_range = 90

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": 'RGB888', "size": (640, 480)}))
picam2.start()

print("Recording... Press Ctrl+C to stop.")
prev = None
start_time = time.time()
while time.time() - start_time < 100:  # Record for 10 seconds
    frame = picam2.capture_array()
    # Display the video feed in a window
    cv2.imshow('Camera Feed', frame)
    if prev is not None:
        mse = np.square(np.subtract(frame, prev)).mean()
        if mse > 7:
            print("Movement!")
                        
            shoulder.angle = 0
            elbow.angle = 0
            time.sleep(.5)
            shoulder.angle = 45
            elbow.angle = 90
            time.sleep(.5)
            shoulder.angle = 0
            elbow.angle = 0
            
            
            
            
        else:
            print("All good")

        
        
    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        picam2.close()
        cv2.destroyAllWindows()  # Close the window
        print("Recording stopped.")
    
    prev = frame


    

