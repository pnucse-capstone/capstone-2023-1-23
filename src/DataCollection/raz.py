import socket
import threading
import os
import time
import smbus
import RPi.GPIO as g

from imusensor.MPU9250 import MPU9250
from imusensor.filters import kalman

# TCP 통신을 위한 socket 설정
ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Raspberry Pi에서 서버 생성
ls.bind(("192.168.2.117", 2222))
ls.listen()
print("waiting for client")
s, clientAddress = ls.accept()
print("connected")

# i2c 통신을 통해 MPU9250 센서와 연결
address = 0x68
bus = smbus.SMBus(1)
imu = MPU9250.MPU9250(bus, address)
imu.begin()

# 기존에 생성해둔 calibration 파일을 사용하여 센서를 교정
imu.loadCalibDataFromFile("/home/pi/test/calib.json")

# Raspberry Pi의 GPIO 5, 6, 12, 13번 핀을 OUTPUT으로 설정
g.setmode(g.BCM)
g.setup(5, g.OUT)
g.setup(6, g.OUT)
g.setup(12, g.OUT)
g.setup(13, g.OUT)

# GPIO 12, 13번 핀을 PWM으로 설정
pwm0 = g.PWM(12, 450)
pwm1 = g.PWM(13, 450)
pwm0.start(0)
pwm1.start(0)

# MPU9250 센서의 노이즈를 줄이기 위해 kalman filter 사용
sensorfusion = kalman.Kalman()

# MPU9250에서 읽어온 초기값을 이후 kalman filter 사용을 위해 sensorfusion에 저장, 필요한 변수 생성
imu.readSensor()
imu.computeOrientation()
sensorfusion.roll = imu.roll
sensorfusion.pitch = imu.pitch
sensorfusion.yaw = imu.yaw
count = 0
currTime = time.time()

# PWM 비율을 결정
# 모터 속도를 조절
sp = 100

while True:
        # rospy에서 TCP 통신을 통해 받은 rm flag를 설정
        temp = s.recv(1024)
        rm = temp.decode()[-1]
        
        # rm이 0인 경우 모터의 움직임 없음
        if rm == "0":
                pwm0.ChangeDutyCycle(0)
                pwm1.ChangeDutyCycle(0)
        else:
        # rm이 0이 아닌 경우 모터 속도는 sp로 설정된 속도만큼 회전
                pwm0.ChangeDutyCycle(sp)
                pwm1.ChangeDutyCycle(sp)
                
                # 1, 2, 3, 4에 따라 모터 방향을 결정
                # 전진과 후진, 양방향 회전
                if rm == "1":
                        g.output(5, False)
                        g.output(6, False)
                elif rm == "2":
                        g.output(5, True)
                        g.output(6, True)
                elif rm == "3":
                        g.output(5, False)
                        g.output(6, True)
                elif rm == "4":
                        g.output(5, True)
                        g.output(6, False)
                else:
                        pwm0.ChangeDutyCycle(0)
                        pwm1.ChangeDutyCycle(0)
                        
        # MPU9250에서 값을 읽어오고 kalman filter를 적용
        imu.readSensor()
        imu.computeOrientation()
        newTime = time.time()
        dt = newTime - currTime
        currTime = newTime
        sensorfusion.computeAndUpdateRollPitchYaw(imu.AccelVals[0], imu.AccelVals[1], imu.AccelVals[2], imu.GyroVals[0], imu.GyroVals[1], imu.GyroVals[2], imu.MagVals[0], imu.MagVals[1], imu.MagVals[2], dt)

        # 계산된 회전각을 rospy에 TCP 통신을 통해 전송
        sm = str(sensorfusion.yaw)
        s.send(sm.encode())
        
s.close()
exit(0)