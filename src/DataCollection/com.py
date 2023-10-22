#! /usr/bin/env python

from re import T
from tf import TransformBroadcaster
import rospy
from rospy import Time
import numpy as np
import math
import socket
import threading
import os
import keyboard
import time

# TCP 통신을 위한 socket 설정
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("waiting for server")
# rospy에서 Raspberry Pi의 서버에 연결
s.connect(("192.168.2.117", 2222))
print("connected")

# transform frame을 다른 노드와 공유하기 위한 broadcaster 노드 설정
rospy.init_node('my_broadcaster')
b = TransformBroadcaster()

# 현재 ROS 내의 회전각이 저장
yaw = 0.0

# TCP 통신에서 전송을 위한 msg 변수
msg = "0"

# transformation frame을 위한 변수
translation = (0.0, 0.0, 0.0)
rotation = (0.0, 0.0, 0.0, 1.0)
tx, ty = 0.0, 0.0
tsp = 0.0045

# 회전각의 한계치
threshold = 35

# 회전각 평균을 구하기 위한 그룹 내의 측정값 개수
count = 0
maxCount = 10

# -180도 주변을 측정할 때 노이즈 발생 감지
# 그룹 내의 노이즈 개수
noiseCount = 0
maxNoiseCount = 3

# 그룹 내의 측정값 총합, 평균 계산
temp = 0

while True:
	# 키보드 입력에 따라 transformation frame 전진, 후진
	# Raspberry Pi에 전달할 카트 전진, 후진 신호 설정
	if keyboard.is_pressed("up"):
		msg = "1"
		tx += tsp * np.cos(yaw * (math.pi / 180)) 
		ty += tsp * np.sin(yaw * (math.pi / 180))
	elif keyboard.is_pressed("down"):
		msg = "2"
		tx -= tsp * np.cos(yaw * (math.pi / 180)) 
		ty -= tsp * np.sin(yaw * (math.pi / 180)) 

	# Raspberry Pi에 전달할 카트 회전 신호 설정
	elif keyboard.is_pressed("left"):
		msg = "3"
	elif keyboard.is_pressed("right"):
		msg = "4"

	elif keyboard.is_pressed("q"):
		break
	
	# 키보드 입력이 없거나 다른 키를 누른 경우는 아무 행동도 하지 않도록 신호 설정
	else:
		msg = "0"

	# 설정된 움직임 신호를 Raspberry Pi에 TCP 통신을 통해 전송
	s.send(msg.encode())

	# translation과 rotation을 변경하여 transform frame에 적용
	translation = (tx, ty, 0.0)

	qx = np.sin(0) * np.cos(0) * np.cos((yaw * (math.pi / 180)) / 2) - np.cos(0) * np.sin(0) * np.sin((yaw * (math.pi / 180)) / 2) 
	qy = np.cos(0) * np.sin(0) * np.cos((yaw * (math.pi / 180)) / 2) + np.sin(0) * np.cos(0) * np.sin((yaw * (math.pi / 180)) / 2) 
	qz = np.cos(0) * np.cos(0) * np.sin((yaw * (math.pi / 180)) / 2) - np.sin(0) * np.sin(0) * np.cos((yaw * (math.pi / 180)) / 2) 
	qw = np.cos(0) * np.cos(0) * np.cos((yaw * (math.pi / 180)) / 2) + np.sin(0) * np.sin(0) * np.sin((yaw * (math.pi / 180)) / 2)
	rotation = (qx, qy, qz, qw)

	b.sendTransform(translation, rotation, Time.now(), 'robot', 'map')

	# 센서 측정값을 받아 temp에 누적
	m = s.recv(1024)
	t = float(m.decode())
	temp += t
	count += 1

	# 전의 yaw 회전각과 비교하여 노이즈인지 판별
	if abs(yaw - t) > threshold and abs((yaw + 360) - t) > threshold and abs(yaw - (t + 360)) > threshold and yaw != 0.0:
		noiseCount += 1

	if count == maxCount:
		# temp를 평균값으로 변경
		temp /= maxCount
		count = 0

		# 노이즈가 많이 생기는 구간이면 회전각을 -180도로 고정
		if noiseCount > maxNoiseCount:
			yaw = -180
		
		# threshold를 넘지 않는 회전각 변화만 적용
		elif abs(yaw - temp) < threshold or abs((yaw + 360) - temp) < threshold or abs(yaw - (temp + 360)) < threshold or yaw == 0.0:
			yaw = temp

		noiseCount = 0
		temp = 0

	time.sleep(0.001)

s.close()
exit(0)