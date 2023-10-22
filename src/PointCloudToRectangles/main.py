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
import warnings
import time
import os
import shutil

#텍스트파일로부터 파라미터값 읽기
def getParameter():
    f = open("parameters.txt")
    result = {}
    nowFunc = ""
    for line in f.readlines():
        words = line.split()
        if words[0] == "#":
            nowFunc = words[1]
            result[nowFunc] = {}
        else:
            if words[1] == "int":
                result[nowFunc][words[0]] = int(words[2])
            else:
                result[nowFunc][words[0]] = float(words[2])

    return result



#파일로부터 점들을 읽어 (n,3) 넘파이 배열로 리턴
def getPoints(fileName):
    f = open(fileName)

    for i in range(11):
        f.readline()

    s = f.read().replace("\n",",")
    s = s.replace(" ",",")
    s = s[:-1]

    f.close()

    points = list(map(float,s.split(",")))

    pointsNp = np.zeros((len(points)//3,3))

    for i in range(0,len(points),3):
        pointsNp[i//3] = [points[i],points[i+1],points[i+2]]

    #return ts.cube
    return pointsNp

def removeFloor(points,floorPoint,threshold,axis):
    axisValues = points[:,axis]
    sortAxisValues= np.sort(axisValues)
    thresholdValue = sortAxisValues[floorPoint] + threshold

    sub = np.zeros((1,3))
    sub[0,axis] = thresholdValue

    return points[np.where(axisValues > thresholdValue)] - sub





# 포인트 그룹을 넘겨주면 각 평면에 속한 포인트 그룹으로 분류하여 리턴
# groupSize : 한번에 RANSAC을 적용하여 평면방정식을 얻을 그룹 크기, 해당 그룹은 랜덤한 한 포인트에 근접한 이웃들로 구성됨
# RANSACloop : RANSAC에서 평면방정식을 얻기 위한 샘플링 횟수
# RANSACSampleSize : RANSAC에서 평면방정식을 얻는데 사용할 기준 포인트 개수
# inlierDis : RANSAC에서 평면방정식으로 inlier기준으로 삼는 평면과의 거리
# mergeDis : RANSAC에서 얻은 inlier에서 유도한 평면방정식으로 전체 포인트에서 해당 평면에 대해 분류할 기준 거리
# minInlier : RANSAC에서 얻은 inlier의 개수가 minInlier미만일시 무시하고 다음 루프 실행
# edgeDis : bfs 클러스터링에서 사용할 bfs edge 거리
# removeSize : 클러스터링으로 분류된 그룹이 removeSize이하 일시 제거
# checkAxisNeighbors : 클러스터링시 각 축에 대해 checkAxisNeighbors만큼 가까운 이웃들을 인접 노드로 간주하여 bfs로 클러스터링한다.
def getPlanesWithClustering(pPoints, groupSize,RANSACloop,RANSACSampleSize,inlierDis,mergeDis,minInlier,edgeDis,removeSize,checkAxisNeighbors,minPlanePoints):
    points = pPoints.copy()
    planes = []
    planePoints = []

    #maxChance = 2 #추출한 평면이 removeSize보다 작을경우 underRemoveSize에 카운팅하며 maxChance번 연속으로 카운팅되면 추출을 종료한다.
    #underRemoveSize = 0

    maxFail = 20 #RANSAC결과가 minInlier보다 작을경우 nowFail에 카운팅하며 maxFail번 연속으로 카운팅되면 minInlier을 절반으로 줄인다.
    nowFail = 0
    planePointsFail = 0
    numMaxFails = 0
    numPlanePointsFail = 0
    print(" ",end='')
    while(len(points) >= groupSize*2): #각 루프에서 평면 그룹 포인트들을 추출한다.
        print("\rremain points :", points.shape[0],"  planes :",len(planes),"  maxFails :",numMaxFails,"  planePointsFails :",numPlanePointsFail, "     ",end='')

        group = al.BFknn(points,random.randint(0,points.shape[0]),min(groupSize,points.shape[0])) #points의 랜덤한 한 점에서 인접한 GroupSize개의 포인트를 가지고온다.
        inlier = al.RANSAC(group,RANSACloop,RANSACSampleSize,inlierDis) #가져온 포인트들에 RANSAC을 적용시켜 한 평면에 대한 inlier를 추출한다.
        if inlier.shape[0] < minInlier: #inlier수가 minInlier보다 작을경우에 대한 처리
            nowFail +=1
            if nowFail == maxFail:
                minInlier /= 2
                nowFail = 0
                numMaxFails+=1
            continue
        nowFail = 0
        plane = al.PointsToPlane(inlier) #inlier들에 선형회귀를 적용하여 평면을 구한다. 평면은 한 점과 법선으로 표현된다.

        (inlierPoints,resultIndex) = al.getInlierOnPlane(points,plane,mergeDis) #points에서 해당 평면에 mergeDis보다 가까운 점들을 추출한다.
        (cPoints, cResultIndex) = al.clusteringWithBFS(inlierPoints,edgeDis,0,checkAxisNeighbors) #추출된 점들에 BFS를 적용하여 클러스터링을 한다.
        maxi = 0
        for i in range(len(cPoints)):
            if cPoints[maxi].shape[0] < cPoints[i].shape[0] :
                maxi = i
        inlierPoints = cPoints[maxi] # 가장 수가 많은 포인트 그룹을 추출한다.

        if inlierPoints.shape[0] < minPlanePoints: #inlier수가 minPlanePoints보다 작을경우에 대한 처리
            planePointsFail +=1
            if planePointsFail == maxFail:
                minPlanePoints /= 2
                mergeDis *= 0.8
                planePointsFail = 0
                numPlanePointsFail+=1
                if minPlanePoints <= removeSize:
                    break
            continue
        planePointsFail = 0


        # if inlierPoints.shape[0] <= removeSize: #최종 추출 포인트 그룹이 removeSize보다 작을경우에 대한 처리
        #     underRemoveSize+=1
        #     if underRemoveSize >= maxChance:
        #         break
        # else:
        #     underRemoveSize = 0
        
        planes.append(plane) #추출한 평면을 결과 평면 배열에 넣는다.

        resultIndex = np.array(resultIndex)
        resultIndex = resultIndex[cResultIndex[maxi]]

        planePoints.append(inlierPoints) # 추출한 포인트 그룹을 결과 포인트 그룹 배열에 넣고 points에서 삭제한다.
        points = np.delete(points,resultIndex,axis=0)

    print()

    planes = np.array(planes)

    warnings.filterwarnings(action="ignore")
    planePoints = np.array(planePoints)
    warnings.filterwarnings(action="default")
    removeIndex = []

    for i in range(planePoints.shape[0]): # 결과에서 removeSize보다 포인트 개수가 작은 포인트 그룹들 삭제
        if planePoints[i].shape[0] < removeSize:
            removeIndex.append(i)
    
    planes = np.delete(planes,removeIndex,axis=0)
    planePoints = np.delete(planePoints,removeIndex,axis=0)
    
    return planes, planePoints



# 평면과 포인트 그룹을 넣으면 각 평면 포인트들을 사각형 네 모서리 포인트로 전환하여 리턴
# numSampleLine : edge points들에서 변 선분을 추출할 때 RANSAC을 통해 얻어낼 후보 선분 개수, 이 선분중 길이가 가장 긴것이 선택된다.
# RANSACloop : edge points들에서 후보 변 선분을 추출할때 반복할 RANSACloop회수, 이중 가장 inleir가 많은 선분이 선택된다.
# n : RANSAC에서 랜덤하게 선택한 n개의 point를 선형회귀하여 직선 방정식을 얻어낸다.
# inlierDis : RANSAC에서 만든 직선 방정식에 대해 inlierDis보다 거리가 작은 point들이 inleir가 된다.
# edgeWidth : convexHull을 통해 얻은 볼록 다각형을 edgeWidth만큼 압축하여 바깥에 있는 점들을 edge Points로 추출한다.
def getRectangles(planes,planePoints,numSampleLine,RANSACloop,n,inlierDis,edgeWidth):
    result = []
    print(' ',end='')
    for i in range(0,len(planes)):
        result.append(al.PlaneToRectangle(planes[i],planePoints[i],numSampleLine,RANSACloop,n,inlierDis,edgeWidth))
        print('\rconverted rectangles :', i,"/",len(planes),end='')
    print()
    return np.array(result)

# 사각형들을 넣으면 유사한 변을 가지며 법선의 각도가 적은 사각형들을 합병하여 리턴
# linePointsDis : 유사한 변을 판별할 때 한 변의 두 점이 상대 변의 두 점에 대해 linePointDis보다 가까우면 유사한 변으로 취급한다.
# limitNormalAngle : 합병 판별 시 법선의 각도가 limitNormalAngle이하여야 한다.(degree)
def mergeRectangles(rectangles,linePointsDis,limitNormalAngle):
    pes = al.rectanglesToPlaneEquations(rectangles) #각 사각형으로부터 평면방정식 추출
    return al.mergeRectangles(rectangles,pes,linePointsDis,limitNormalAngle) # 각 사각형 중 맞닿은 사각형을 합병


def outputResult(rectangles,timeString):

    nowTime = time
    fileName = timeString + "result_" + nowTime.strftime('%Y%m%d_%H%M%S') + ".txt"
    f = open(fileName,"a")

    outputString = ""

    for rect in rectangles:
        for i in range(4):
            for j in range(3):
                outputString += str(rect[i][j]) + ","
        outputString = outputString[:-1]
        outputString += "\n"

    f.write(outputString)
    f.close()

    shutil.copyfile("parameters.txt", timeString+"parameters.txt")






def main():
    ############# 파일 이름 입력 ##############

    fileName = "inputFiles\\" + input("input file name:")
    

    ############# 출력 모드 입력 #####################
    vMode = int(input("1. show all step with matplotlib  2. show only result with matplotlib \n3. show all step with matlab      4. show only result with matlab\n"))
    
    if vMode >=2:
        eng = matlab.engine.start_matlab()

    ############# 파라미터 읽기 #####################

    parameters = getParameter()

    tryCount = int(input("input repeat count:"))

    for tryIndex in range(tryCount):

        ############ 저장 폴더 생성 #####################

        nowTime = time
        timeString = "results\\" + nowTime.strftime('%Y%m%d_%H%M%S')
        os.mkdir(os.getcwd()  + "\\"+ timeString)
        timeString += "\\"

        ############# 파일에서 포인트 읽기 ##################


        points = getPoints(fileName)
        if points.shape[0] > parameters["getPoints"]["numPoints"] :
            points = points[np.unique(np.random.randint(0,points.shape[0],parameters["getPoints"]["numPoints"]))]
        
        print("num of points",len(points))
        if vMode == 1:
            pl.showPoints(points)
        elif vMode == 3:
            pl.showPointsWithMatlab(eng,points,timeString,1)


        #############포인트 threshold#############이후 제거

        axisValues = points[:,2]
        thresholdValue = 0.2#10

        points = points[np.where(axisValues < thresholdValue)]

        print("num of points",len(points))
        if vMode == 1:
            pl.showPoints(points)
        elif vMode == 3:
            pl.showPointsWithMatlab(eng,points,timeString,2)

        ############# 바닥 제거 ##################

        points = removeFloor(points,parameters["removeFloor"]["floorPoint"],parameters["removeFloor"]["threshold"],parameters["removeFloor"]["axis"])

        print("num of points",len(points))
        if vMode == 1:
            pl.showPoints(points)
        elif vMode == 3:
            pl.showPointsWithMatlab(eng,points,timeString,3)


        ############# 최초 평면 그룹 분류 ##############

        (planes, planePoints) = getPlanesWithClustering(points,parameters["pointsClustering"]["groupSize"],parameters["pointsClustering"]["RANSACloop"],parameters["pointsClustering"]["RANSACSampleSize"],parameters["pointsClustering"]["inlierDis"],parameters["pointsClustering"]["mergeDis"],parameters["pointsClustering"]["minInlier"],parameters["pointsClustering"]["edgeDis"],parameters["pointsClustering"]["removeSize"],parameters["pointsClustering"]["checkAxisNeighbors"],parameters["pointsClustering"]["minPlanePoints"])
        print("num of planes:",len(planes))
        if vMode == 1:
            pl.showPointsGroups(planePoints)
        elif vMode == 3:
            pl.showPointsGroupsWithMatlab(eng,planePoints,timeString,4)

        ############# 각 평면 포인터 그룹을 사각형으로 전환 ################

        rectangles = getRectangles(planes,planePoints,parameters["getRectangles"]["numSampleLine"],parameters["getRectangles"]["RANSACloop"],parameters["getRectangles"]["n"],parameters["getRectangles"]["inlierDis"],parameters["getRectangles"]["edgeWidth"])
        if vMode == 1:
            pl.showRectangles(rectangles)
        elif vMode == 3:
            pl.showRectanglessWithMatlab(eng,rectangles,timeString,5)

        ############# 각 사각형들을 합병 #############################

        rectangles = mergeRectangles(rectangles,parameters["mergeRectangles"]["linePointsDis"],parameters["mergeRectangles"]["limitNormalAngle"])
        print("num of merged Rectangles:",rectangles.shape[0])
        if vMode <=2:
            pl.showRectangles(rectangles)
        else:
            pl.showRectanglessWithMatlab(eng,rectangles,timeString,6)

        ############# 최종 결과 텍스트 파일로 출력 ######################

        outputResult(rectangles,timeString)
        


main()