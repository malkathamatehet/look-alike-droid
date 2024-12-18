import board
import busio
import adafruit_pca9685
i2c = busio.I2C(board.SCL, board.SDA)
hat = adafruit_pca9685.PCA9685(i2c)
import time

from adafruit_servokit import ServoKit
kit = ServoKit(channels=16)

shoulder = kit.servo[0]
elbow = kit.servo[1]

shoulder.actuation_range = 45
elbow.actuation_range = 90

shoulder.angle = 0
elbow.angle = 0
time.sleep(.5)
shoulder.angle = 45
elbow.angle = 90
time.sleep(.5)
shoulder.angle = 0
elbow.angle = 0

