import numpy as np
from numpy import random
import math
import plotLib as pl
import util.before as bf
from collections import deque

#브루트 포스로 index에 인접한 k개 이웃 리턴
def BFknn(points,index,k):
    dis = np.square(points - np.array(points[index])).sum(axis=1)
    
    result = points[np.argsort(dis)]

    return result[:k]   

#Point들로 생성된 평면 리턴(평면의 한점, 법선 벡터)
def PointsToPlane(points):
        A = np.c_[points,np.ones((points.shape[0],1))]
        eval, evec = np.linalg.eig(A.T @ A) 
        evec = evec.T 
        planeE = evec[np.argmin(eval)]
        planePoint = np.zeros(3)
        if (planeE[0] != 0):
            planePoint = np.array([-planeE[3]/planeE[0],0,0])
        elif (planeE[1] != 0):
            planePoint = np.array([0,-planeE[3]/planeE[1],0])
        else:
            planePoint = np.array([0,0,-planeE[3]/planeE[2]])

        normal = np.array(planeE[:3])
        normal = normal / np.linalg.norm(normal)
        return (planePoint,normal)
     
#포인트 그룹(count,3)을 넣으면 한 평면을 이루는 inlier그룹을 리턴
def RANSAC(group,loop,n,inlierDis):
    maxGroup = np.zeros((1,3))
    maxNumInlier = 0

    for i in range(loop): #각 루프에서 한 평면을 추출하고 inlier의 개수가 가장 많은 평면을 찾는다.
        indexs = random.randint(group.shape[0],size=n)
        tmpGroup = group[indexs[0]].reshape(1,3)
        for j in range(1,len(indexs)):
            tmpGroup = np.r_[tmpGroup,group[indexs[j]].reshape(1,3)]

        PlanePoints, normal = PointsToPlane(tmpGroup)

        upVector = np.array([0,0,1])

        und = np.dot(upVector, normal)

        if und > 0.1 or -0.1 > und:
            continue
        
        (tmpInGroup,resultIndex) = getInlierOnPlane(group,PointsToPlane(tmpGroup),inlierDis) # 랜덤한 n개의 포인트로 선형회귀 평면을 만들고 inlier를 리턴한다.

        if(maxNumInlier < tmpInGroup.shape[0]): #inlier개수가 가장 많으면 결과로 갱신한다.
            maxNumInlier = tmpInGroup.shape[0]
            maxGroup = tmpInGroup

    return maxGroup

#포인트들중 평면과 inlierDis이하 거리인 포인트들을 리턴
def getInlierOnPlane(points,plane,inlierDis):
    lenNormal = np.linalg.norm(plane[1])

    dis = np.abs(((points - plane[0]) @ plane[1].reshape(3,1)) / lenNormal).reshape(points.shape[0])
    resultIndex = list(np.where(dis <= inlierDis)[0])

    return points[resultIndex], resultIndex


#포인터들을 edgeDis거리로 DFS하여 클러스터링하고 분류된 포인터 그룹들을 리턴
def clusteringWithBFS(points,edgeDis,removeSize,checkAxisNeighbors):
    result = []
    resultIndex = []

    remainIndex = set(range(points.shape[0]))
    sx = np.argsort(points[:,0])# 각 축으로 정렬된 포인트 배열을 생성
    sy = np.argsort(points[:,1])
    sz = np.argsort(points[:,2])

    tx = np.zeros(points.shape[0])
    ty = np.zeros(points.shape[0])
    tz = np.zeros(points.shape[0])

    for i in range(points.shape[0]):# 각 포인트들이 각 축에 대한 정렬 배열에서 몇번째 index에 위치하는지 기록된 배열을 생성
        tx[sx[i]] = i
        ty[sy[i]] = i
        tz[sz[i]] = i

    while len(remainIndex) > 0: #각 루프에서 인접한 점들을 이웃으로하여 BFS를 돌려 한 클러스터링 그룹을 추출
        startIndex = next(iter(remainIndex))
        groupSet = set()
        BFSWithDis(points,startIndex,sx,sy,sz,tx,ty,tz,groupSet,edgeDis,checkAxisNeighbors) # 각 축마다 인접한 checkAxisNeighbors만큼의 점들에 대해 edgeDis보다 근접할시 이웃으로 보고 BFS를 수행 
        remainIndex = remainIndex - groupSet
        if len(groupSet) > removeSize:
            result.append(points[list(groupSet)])
            resultIndex.append(list(groupSet))


    return result, resultIndex


#clusteringWithDFS에서 사용하는 BFS 재귀함수 - private
def BFSWithDis(points,startIndex,sx,sy,sz,tx,ty,tz,groupSet,edgeDis,checkAxisNeighbors):
    checkAxisNeighbors //= 2
    frontier = deque([startIndex])
    groupSet.add(startIndex)

    while(len(frontier) > 0):
        nowIndex = frontier.popleft()
        addChildsWithSortedAxis(points,nowIndex,sx,tx,frontier,groupSet,edgeDis,checkAxisNeighbors)
        addChildsWithSortedAxis(points,nowIndex,sy,ty,frontier,groupSet,edgeDis,checkAxisNeighbors)
        addChildsWithSortedAxis(points,nowIndex,sz,tz,frontier,groupSet,edgeDis,checkAxisNeighbors)

        
#index 포인트를 한 축으로 정렬했을때 앞과 뒤 각각 k개씩의 포인트들에 대해 인접한지 검사하고 groupSet과 Frontier에 삽입
def addChildsWithSortedAxis(points,index,s,t,frontier,groupSet,edgeDis,k):
    si = int(t[index])

    for i in range(max(si-k,0),min(si+k+1,points.shape[0])):
        childIndex = int(s[i])
        if childIndex not in groupSet and GetDis(points,index,childIndex) < edgeDis:
            groupSet.add(childIndex)
            frontier.append(childIndex)

#points의 i,j번째 포인트들의 거리를 리턴
def GetDis(points,i,j):
    if i>=j:
        return np.linalg.norm(points[i] - points[j])
    else:
        return np.linalg.norm(points[j] - points[i])


    

#points들을 plane에 투영하여 리턴
def projectionPointsToPlane(points,plane):
    return points - np.dot(points - plane[0],plane[1]).reshape(len(points),1) * plane[1]

#plane에 투영된 points와 plane을 넘겨주면 plane 법선을 z축으로 하고 xy축은 그에 대한 정규직교인 basis를 반환
def getBasisPointsTo2degreeOnPlane(points,plane):
    basis = np.zeros((3,3))
    basis[0] = points[0] - plane[0]
    
    index = 1
    while np.linalg.norm(basis[0]) == 0:
        basis[0] = points[index] - plane[0]
        index+=1

    basis[0] = basis[0] / np.linalg.norm(basis[0])
    basis[1] = np.cross(basis[0],plane[1])
    basis[2] = plane[1]
    return basis #points@np.linalg.inv(basis) : e->basis

#ccw인지 체크하여 true false 리턴
def ccw(a,b,c,isForCWConvexHull = False):
    if isForCWConvexHull:
        return (b[0] - a[0])*(c[1] - a[1]) - (c[0] - a[0])*(b[1] - a[1]) >= 0.0001
    else:
        return (b[0] - a[0])*(c[1] - a[1]) - (c[0] - a[0])*(b[1] - a[1]) >= 0.0001

#points들을 xy기준으로 convexHull을 수행, cw true시 시계방향 순서 포인트를 반환, false시 반 시계방향
def convexHull(points,cw):
    points = points.copy()
    startIndex = np.argmin(points,axis=0)[1]
    start = points[startIndex]
    np.delete(points,startIndex,axis=0)
    angles = []
    for p in points:
        p2s = p-start
        if p2s[0] == 0 :
            angle=math.pi/2
        else:
            angle = math.atan(p2s[1]/p2s[0])
        angles.append(angle if angle>=0 else math.pi + angle)
    points = points[np.argsort(angles)]
    if cw:
        points = points[::-1]
    result = [start,points[0]]
    for i in range(1,len(points)):
        while cw == (ccw(result[-2],result[-1],points[i],cw)):
            if (len(result)==2):
                break
            result = result[:-1]
        result.append(points[i])
    return np.array(result)

#points들의 xy값을 선형회귀하여 2차원 직선 방정식 a,b,c 리턴 
def pointsToLine(points):
    if np.array_equal(points[0],points[1]):
        return np.array([1,1,1])
    A = np.c_[points[:,:-1],np.ones((points.shape[0],1))]
    eval, evec = np.linalg.eig(A.T @ A) 
    evec = evec.T 
    return evec[np.argmin(eval)]

# 포인트들과 그 포인트에 대한 convexHull의 결과를 넘겨주면 포인트가 이루는 형상의 edge 부근 포인트를 반환
# convexHull이 cw순이면 cw에 true를, 반대면 false를 넘겨주어야함.
def getEdgePointsWithConvex(points,ConvexPoints,reduceLength,cw):
    middlePoint = np.sum(ConvexPoints,axis=0) / ConvexPoints.shape[0]
    c2m = (middlePoint - ConvexPoints)
    c2m = c2m / np.linalg.norm(c2m,axis=1).reshape((c2m.shape[0],1))
    ConvexPoints = ConvexPoints + (c2m * reduceLength)
    result = []
    for p in points:
        for i in range(0,ConvexPoints.shape[0]-1):
            if cw == (ccw(ConvexPoints[i],ConvexPoints[i+1],p)):
                result.append(p)
                break
    return np.array(result)

#points들이 이루는 형상의 edge부근 포인트들을 반환
def getEdgePoints(points , reduceLength):
    convexPoints = points
    while(True): # 오차 극복을 위해 반복적으로 convex hull 적용
        convexPoints = convexHull(convexPoints,False)
        numFirst = convexPoints.shape[0]
        convexPoints = convexHull(convexPoints,True)
        if numFirst <= convexPoints.shape[0]:
            break
    
    points = getEdgePointsWithConvex(points ,convexPoints,reduceLength,True) #convex hull의 결과를 중앙으로 reduceLength만큼 모으고 그 밖의 점들을 edgePoints로 추출

    if points.shape[0] < convexPoints.shape[0]:
        return convexPoints

    return points


#points들에 RANSAC을 적용하여 얻은 높은 inlier를 가진 직선중 가장 긴 직선을 반환
def getBestLineInEdgePoints(points,numSampleLine,RANSACloop,n,inlierDis):
    
    maxLength = 0
    bestLine = np.zeros((1,3))

    pointXs = points[:,0]
    pointYs = points[:,1]


    for i in range(numSampleLine): # 각 loop에서 RANSAC으로 선을 얻고 그 중 길이가 가장 긴 것을 갱신 
        maxInliers = np.zeros((1,3))
        maxline = np.zeros((1,3))
        for j in range(RANSACloop): # RANSACloop를 통해 inlier가 가장 많은 직선을 추출
            choiceI = random.randint(0,points.shape[0],size=n)
            linePoints = np.zeros((1,3))
            for i in choiceI:
                linePoints = np.r_[linePoints,points[i].reshape(1,3)]
            line = pointsToLine(linePoints[1:])
            nowInliers = np.zeros((1,3))
            under = (line[0]**2 + line[1]**2)**0.5
            

            nowInliers = np.r_[nowInliers ,points[list(np.where( (np.abs(line[0]*pointXs + line[1]*pointYs + line[2]) / under) <=inlierDis)[0])]]
           
            if nowInliers.shape[0] > maxInliers.shape[0]:
                maxInliers = nowInliers
                maxline = line

        #RANSAC으로 얻은 라인의 길이 구하기
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
        if maxLength < nowLength: #가장 길다면 갱신
            maxLength = nowLength
            bestLine = maxline

    return bestLine
            


#plane과 plane을 이루는 points들을 입력 시 해당 points들의 사각형을 추측하여 네 모서리점을 반환
def PlaneToRectangle(plane,points,numSampleLine,RANSACloop,n,inlierDis,edgeWidth): 
    points = projectionPointsToPlane(points,plane) #포인트 그룹을 평면에 투영

    allPoints = points
    basisForConvexHull = getBasisPointsTo2degreeOnPlane(points,plane) #평면의 법선을 z축으로 하는 정규 직교 기저 행렬을 계산
    points = points@np.linalg.inv(basisForConvexHull) # convex hull 적용을 위해 해당 기저로 좌표변환


    points = getEdgePoints(points,edgeWidth) #convex hull들의 알고리즘을 이용하여 edge points를 추출

    
    line = getBestLineInEdgePoints(points,numSampleLine,RANSACloop,n,inlierDis) # edge points에 RANSAC을 적용하여 최적의 직선 방정식을 추출
    
    
    lineVector = np.array([1,-line[0]/line[1],0])
    lineVector = lineVector @ basisForConvexHull
    lineVector = lineVector/np.linalg.norm(lineVector)
    basisForRectangle = np.array([lineVector,np.cross(lineVector,plane[1]),plane[1]]) # 해당 직선 방향의 x축, 법선 방향의 z축을 가지는 정규 직교 기저 행렬을 계산


    allPoints = allPoints@np.linalg.inv(basisForRectangle) #사각형의 모서리 네 포인트를 추출하기 위해 해당 기저로 좌표변환

    maxs = np.max(allPoints,axis = 0)
    mins = np.min(allPoints,axis = 0)
    minX = mins[0]
    maxX = maxs[0]
    minY = mins[1]
    maxY = maxs[1]

    RectanglePoints = np.array([[minX,minY,allPoints[0][2]],
                               [maxX,minY,allPoints[0][2]],
                               [maxX,maxY,allPoints[0][2]],
                               [minX,maxY,allPoints[0][2]]]) #각 포인트들의 최대, 최소 XY값으로 사각형의 네 점 추출
    return RectanglePoints @ basisForRectangle # 표준 기저로 좌표 변환하여 리턴

#사각형과 한 라인이 주어지면 각 라인을 이루는 두 점이 근접한 변의 인덱스를 리턴한다.
def getMatchLine(rectangle,line,linePointDis):
    for i in range(0,4):
        if  np.linalg.norm(rectangle[i] - line[0]) <= linePointDis and np.linalg.norm(rectangle[(i+1)%4] - line[1]) <= linePointDis:
            return i
        elif np.linalg.norm(rectangle[i] - line[1]) <= linePointDis and np.linalg.norm(rectangle[(i+1)%4] - line[0]) <= linePointDis:
            return i
    return -1

#사각형들을 평면방정식들로 변환하여 리턴한다.
def rectanglesToPlaneEquations(rectangles):
    cr = np.cross(rectangles[:,1]-rectangles[:,0],rectangles[:,2]-rectangles[:,0],axis=1)
    cr = cr/np.linalg.norm(cr,axis=1).reshape(cr.shape[0],1)
    d = -np.sum(rectangles[:,0] * cr,axis=1)
    return np.c_[cr.reshape(cr.shape[0],3),d.reshape(d.shape[0],1)]

#사각형을 평면방정식으로 변환하여 리턴한다.
def rectangleToPlaneEquation(rectangle):
    cr = np.cross(rectangle[1]-rectangle[0],rectangle[2]-rectangle[0])
    cr = cr/np.linalg.norm(cr)
    d = -np.dot(rectangle[0],cr)
    return np.array([cr[0],cr[1],cr[2],d])

#사각형을 이루는 네개의 점을 입력하면 적절한 순서로 변경하여 사각형으로 리턴한다.
def fourPointsToRectangles(points):
    sp = points[np.linalg.norm(points - points[0],axis=1).argsort()]
    return np.array([sp[0],sp[1],sp[3],sp[2]])

#평면의 두 법선을 입력하면 평면사이의 각도를 degree로 리턴한다.
def getAngleTwoNormal(a,b):
    r = np.arccos(np.dot(a,b) / (np.linalg.norm(a) *np.linalg.norm(b)))
    d = r *(180/math.pi)
    if d >= 90 :
        return 180-d
    else:
        return d

#한 변이 맞다은 두 사각형을 근사적으로 합병하여 리턴한다.
def mergeTwoRectangles(ra,rb):
    basis = np.zeros((3,3))
    basis[0] = ra[1] - ra[0]
    basis[0] = basis[0]/np.linalg.norm(basis[0])
    basis[1] = ra[3] - ra[0]
    basis[1] = basis[1]/np.linalg.norm(basis[1])
    basis[2] = np.cross(basis[0],basis[1])
    ibasis = np.linalg.inv(basis)
    pts = np.r_[ra,rb]@ibasis

    maxs = np.max(pts,axis=0)
    mins = np.min(pts,axis=0)
    minX = mins[0]
    maxX = maxs[0]
    minY = mins[1]
    maxY = maxs[1]

    RectanglePoints = np.array([[minX,minY,pts[0][2]],
                               [maxX,minY,pts[0][2]],
                               [maxX,maxY,pts[0][2]],
                               [minX,maxY,pts[0][2]]])
    return RectanglePoints@basis

#사각형과 그에 대한 평면방정식들을 입력하면 맞다은 사각형들을 합병하여 리턴한다.
def mergeRectangles(rectangles,planeEquations,linePointDis,limitNormalAngle):
    result = []
    resultPlaneEquations = []
    rectangles = rectangles.copy()
    planeEquations = planeEquations.copy()

    while rectangles.shape[0] > 1: #각 루프에서 합병된 사각형을 추출
        r = rectangles[0] # 랜덤한 한 사각형 추출 및 rectangles에서 제외
        rectangles = rectangles[1:]
        rp = planeEquations[0]
        planeEquations = planeEquations[1:]

        isMerged=False
        isE = False
        while True: # 더 이상 합병이 불가능할 때까지 반복
            if isE: # isE == true면 더이상 합병이 불가능
                break
            
            isC = False
            for i in range(0,4): # 선택된 사각형의 각 변에 대해 맞다은 사각형을 찾아 합병 시도
                if isC:
                    break
                for ri in range(rectangles.shape[0]): # 각 사각형에 합병 시도
                    testR = rectangles[ri]
                    if getAngleTwoNormal(rp[:-1],planeEquations[ri][:-1]) <= limitNormalAngle: # 두 사각형 평면의 각도 검사
                        ml = getMatchLine(testR,(r[i],r[(i+1)%4]),linePointDis) # 현재 검사 중인 변과 사각형의 각 변 중 유사한 변이 있는지 검사
                        if ml != -1: # 유사한 변이 있다면 합병
                            rectangles = np.delete(rectangles,ri,axis=0)
                            planeEquations = np.delete(planeEquations,ri,axis=0)
                            r = mergeTwoRectangles(r,testR) # 두 사각형을 합병하고 합병된 사각형으로 다시 반복하며 합병시도
                            isMerged = True
                            isC = True
                            break
                if isMerged and not isC: #합병되어 새로 나온 사각형이라면 기존에 합병됬던 사각형에 대해서도 추가적인 합병 검사
                    for ri in range(len(result)):
                        testR = result[ri]
                        if getAngleTwoNormal(rp[:-1],resultPlaneEquations[ri][:-1]) <= limitNormalAngle:
                            ml = getMatchLine(testR,(r[i],r[(i+1)%4]),linePointDis)
                            if ml != -1:
                                del result[ri]
                                del resultPlaneEquations[ri]
                                r = mergeTwoRectangles(r,testR) 
                                isMerged = True
                                isC = True
                                break
            if not isC:    # 추가적인 합병이 없다면 결과에 사각형을 넣고 다음 합병 루프로 넘어감
                result.append(r)
                resultPlaneEquations.append(rectangleToPlaneEquation(r))
                isE = True
                break 
    
    if rectangles.shape[0] > 0:
        result.append(rectangles[0])
    
    return np.array(result)

            






    

    
    

        



        
