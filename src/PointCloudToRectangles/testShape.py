import numpy as np

def makeCube(cubeD,cubeLineSize):
    cubeSurfaceSize = cubeLineSize**2
    cube = np.zeros((cubeSurfaceSize*6,3))
    index = 0

    for y in range(0,cubeLineSize):
        for x in range(0,cubeLineSize):
            cube[index] = [x*cubeD,y*cubeD,0]
            cube[index+cubeSurfaceSize] = [x*cubeD,y*cubeD,(cubeLineSize-1)*cubeD]
            cube[index+cubeSurfaceSize*2] = [x*cubeD,0,y*cubeD] 
            cube[index+cubeSurfaceSize*3] = [x*cubeD,(cubeLineSize-1)*cubeD,y*cubeD] 
            cube[index+cubeSurfaceSize*4] = [0,x*cubeD,y*cubeD] 
            cube[index+cubeSurfaceSize*5] = [(cubeLineSize-1)*cubeD,x*cubeD,y*cubeD] 
            index+=1
            
    return cube

def makeWall(cubeD,cubeLineSize):
    cubeSurfaceSize = cubeLineSize**2
    cube = np.zeros((cubeSurfaceSize*6,3))
    index = 0

    cubeZD = cubeD * 2 / 5

    for y in range(0,cubeLineSize):
        for x in range(0,cubeLineSize):
            cube[index+cubeSurfaceSize*2] = [x*cubeD,0,y*cubeZD] 
            cube[index+cubeSurfaceSize*3] = [x*cubeD,(cubeLineSize-1)*cubeD,y*cubeZD] 
            cube[index+cubeSurfaceSize*4] = [0,x*cubeD,y*cubeZD] 
            cube[index+cubeSurfaceSize*5] = [(cubeLineSize-1)*cubeD,x*cubeD,y*cubeZD] 
            index+=1
            
    return cube



cubeD = 0.05
cubeLineSize = 100

cube = makeCube(cubeD,cubeLineSize)

cube2 = makeCube(cubeD,cubeLineSize) + np.array([cubeD * cubeLineSize * 2,cubeD * cubeLineSize * 2,0])

cube3 = makeCube(cubeD,cubeLineSize) + np.array([cubeD * cubeLineSize * 4,cubeD * cubeLineSize * 2,0])

wall = makeWall(cubeD*2.5,cubeLineSize * 2)

cube = np.r_[cube,cube2,cube3,wall]