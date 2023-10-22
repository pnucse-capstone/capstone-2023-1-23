import plotLib as pl
import algorithmLib as al
import testShape as ts
import util.before as bf
import matplotlib.pyplot as plt
import numpy as np
import time
from numpy import random
import matlab.engine
from matlab import double as double_m

# 포인트 그룹을 넘겨주면 각 평면에 속한 포인트 그룹으로 분류하여 리턴
# groupSize : 한번에 RANSAC을 적용하여 평면방정식을 얻을 그룹 크기, 해당 그룹은 랜덤한 한 포인트에 근접한 이웃들로 구성됨
# RANSACloop : RANSAC에서 평면방정식을 얻기위한 샘플링 횟수
# RANSACSampleSize : RANSAC에서 평면방정식을 얻는데 사용할 기준 포인트 개수
# inlierDis : RANSAC에서 평면방정식으로 inlier기준으로 삼는 평면과의 거리
# mergeDis : RANSAC에서 얻은 inlier에서 유도한 평면방정식으로 전체 포인트에서 해당 평면에 대해 분류할 기준 거리
# minInlier : RANSAC에서 얻은 inlier의 개수가 minInlier미만일시 무시하고 다음 루프 실행 
def getPlanes(pPoints, groupSize,RANSACloop,RANSACSampleSize,inlierDis,mergeDis,minInlier):
    points = pPoints.copy()
    planes = []
    planePoints = []

    maxFail = 20
    nowFail = 0

    while(points.shape[0]>=groupSize/5):
         group = al.BFknn(points,random.randint(0,points.shape[0]),min(groupSize,points.shape[0]))
         inlier = al.RANSAC(group,RANSACloop,RANSACSampleSize,inlierDis)
         if inlier.shape[0] < minInlier:
            nowFail +=1
            if nowFail == maxFail:
                minInlier /= 2
                nowFail = 0
                print(len(planes),"th plane" ," maxFail!")
            continue
         nowFail = 0
         plane = al.PointsToPlane(inlier)
         planes.append(plane)
         (inlierPoints,resultIndex) = al.getInlierOnPlane(points,plane,mergeDis)
         planePoints.append(inlierPoints)
         points = np.delete(points,resultIndex,axis=0)
    
    return planes, planePoints


# getPlanes에서 얻은 평면과 포인트 그룹을 넣으면 각 평면 포인트 그룹에 클러스터링을 적용하여 인접한 포인트 그룹으로 분리하여 리턴
# edgeDis : dfs 클러스터링에서 사용할 dfs edge 거리
# removeSize : 클러스터링으로 분류된 그룹이 removeSize이하 일시 제거
def getClusteredPlanes(planes,planePoints,edgeDis,removeSize,checkAxisNeighbors):
    resultPlanes = []
    resultPlanePoints = []
    for i in range(0,len(planes)):
        (clusteringResult, resultIndex) = al.clusteringWithBFS(planePoints[i],edgeDis,removeSize,checkAxisNeighbors)
        for j in range(0,len(clusteringResult)):
            resultPlanes.append(planes[i])
        resultPlanePoints += clusteringResult
    return resultPlanes, resultPlanePoints



def main():
    pass
    ############# 각 평면 그룹에 클러스터링 적용 ################

    """
    (planes, planePoints) = getClusteredPlanes(planes,planePoints,0.1,13,10)
    print("num of clustered planes:",len(planes))
    if vMode == 1:
        pl.showPointsGroups(planePoints)
    if vMode == 3:
        pl.showPointsGroupsWithMatlab(eng,planePoints)
        input("press enter for next step")
    """

    ############# 각 평면 포인터 그룹을 사각형으로 전환 ################
