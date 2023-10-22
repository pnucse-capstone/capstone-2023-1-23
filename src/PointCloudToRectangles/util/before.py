import plotLib as pl
import algorithmLib as al
import testShape as ts
import matplotlib.pyplot as plt
import numpy as np
from numpy import random

def BFknn(points,index,k):
    dis = np.square(points - np.array(points[index])).sum(axis=1)
    
    maxInit = 10000000
    mins = []

    for i in range(0,k):
        mins.append((-1,maxInit))
    
    for i in range(0,points.shape[0]):
        if i == index:
            continue
        result = 0
        for j in range(0,k):
            if(mins[j][1] > dis[i]):
                result+=1
            else:
                break
        if result > 0:
            mins.insert(result,(i,dis[i]))
            mins = mins[1:]
    result = np.array([[0,0,0]])
    for i in mins:
        result = np.r_[result,np.array([points[i[0]]])]

    return result[1:]


def getInlierOnPlane(points,plane,inlierDis):
    lenNormal = np.linalg.norm(plane[1])
    result = np.zeros((1,3))
    resultIndex = []

    index = 0

    for nowPoint in points:
        if inlierDis >= np.linalg.norm(np.dot(nowPoint-plane[0],plane[1])) / lenNormal:
            result = np.r_[result,nowPoint.reshape(1,3)]
            resultIndex.append(index)
        index+=1
    return result[1:],resultIndex


def projectionPointsToPlane(points,plane):
    result = np.zeros(points.shape)
    
    for i in range(0,points.shape[0]):
        result[i] = points[i] - (np.dot(points[i] - plane[0],plane[1]) * plane[1])

    return result


def clusteringWithDFS(pPoints,edgeDis,removeSize):
    points = pPoints.copy()
    result = []

    while points.shape[0] > 0:
        groupSet = set()
        DFSWithDis(points,0,groupSet,edgeDis)
        if len(groupSet) > removeSize:
            tmpPoints = np.zeros((1,3))
            for i in groupSet:
                tmpPoints = np.r_[tmpPoints,points[i].reshape(1,3)]
            result.append(tmpPoints[1:])
        points = np.delete(points,list(groupSet),axis=0)
    
    return result


#clusteringWithDFS에서 사용하는 DFS 재귀함수 - private
def DFSWithDis(points,index,groupSet,edgeDis):
    groupSet.add(index)
    for i in range(0,points.shape[0]):
        if i not in groupSet and np.linalg.norm(points[index]-points[i]) <= edgeDis:
            DFSWithDis(points,i,groupSet,edgeDis)


def getBestLineInEdgePoints(points,numSampleLine,RANSACloop,n,inlierDis):
    
    maxLength = 0
    bestLine = np.zeros((1,3))

    for i in range(numSampleLine):
        maxInliers = np.zeros((1,3))
        maxline = np.zeros((1,3))
        for j in range(RANSACloop):
            choiceI = random.randint(0,points.shape[0],size=n)
            linePoints = np.zeros((1,3))
            for i in choiceI:
                linePoints = np.r_[linePoints,points[i].reshape(1,3)]
            line = pointsToLine(linePoints[1:])
            nowInliers = np.zeros((1,3))
            under = (line[0]**2 + line[1]**2)**0.5
            for p in points:
                if inlierDis >= abs(line[0]*p[0] + line[1]*p[1] + line[2]) / under:
                    nowInliers = np.r_[nowInliers,p.reshape(1,3)]
            if nowInliers.shape[0] > maxInliers.shape[0]:
                maxInliers = nowInliers
                maxline = line

        #maxLine 길이 구하기
        minP = maxInliers[1]
        maxP = maxInliers[1]
        if -1 <= -maxline[0]/maxline[1] <= 1:
            for k in range(2,maxInliers.shape[0]):
                minP = minP if minP[0] < maxInliers[k][0] else maxInliers[k]
                maxP = maxP if maxP[0] > maxInliers[k][0] else maxInliers[k]
        else:
            for k in range(2,maxInliers.shape[0]):
                minP = minP if minP[1] < maxInliers[k][1] else maxInliers[k]
                maxP = maxP if maxP[1] > maxInliers[k][1] else maxInliers[k]
        nowLength = np.linalg.norm(minP-maxP)
        if maxLength < nowLength:
            maxLength = nowLength
            bestLine = maxline

    return bestLine

def PlaneToRectangle(plane,points,numSampleLine,RANSACloop,n,inlierDis,edgeWidth): 
    points = projectionPointsToPlane(points,plane)
    allPoints = points
    basisForConvexHull = getBasisPointsTo2degreeOnPlane(points,plane)
    points = points@np.linalg.inv(basisForConvexHull)

    points = getEdgePoints(points,edgeWidth)
    
    line = getBestLineInEdgePoints(points,numSampleLine,RANSACloop,n,inlierDis)
    
    
    lineVector = np.array([1,-line[0]/line[1],0])
    lineVector = lineVector @ basisForConvexHull
    lineVector = lineVector/np.linalg.norm(lineVector)
    basisForRectangle = np.array([lineVector,np.cross(lineVector,plane[1]),plane[1]])


    allPoints = allPoints@np.linalg.inv(basisForRectangle)
    minX = allPoints[0][0]
    maxX = allPoints[0][0]
    minY = allPoints[0][1] 
    maxY = allPoints[0][1]
    for p in allPoints[1:]:
        minX = minX if minX < p[0] else p[0]
        maxX = maxX if maxX > p[0] else p[0]
        minY = minY if minY < p[1] else p[1]
        maxY = maxY if maxY > p[1] else p[1]
    RectanglePoints = np.array([[minX,minY,allPoints[0][2]],
                               [maxX,minY,allPoints[0][2]],
                               [maxX,maxY,allPoints[0][2]],
                               [minX,maxY,allPoints[0][2]]])
    return RectanglePoints @ basisForRectangle


