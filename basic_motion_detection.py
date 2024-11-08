from picamera2 import Picamera2
import cv2
import time
import numpy as np


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
            
            
            
        else:
            print("All good")

        
        
    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        picam2.close()
        cv2.destroyAllWindows()  # Close the window
        print("Recording stopped.")
    
    prev = frame


    
