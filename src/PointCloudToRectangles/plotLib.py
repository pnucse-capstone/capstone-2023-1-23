import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from numpy import random
import matlab.engine
from matlab import double as double_m


axisMin = -1
axisMax = 1


def addSurface(plot,plane,xScope,yScope,color=[-1]):
    point  = plane[0]
    normal = plane[1]

    d = -point.dot(normal)

    xx, yy = np.meshgrid(xScope, yScope)

    z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

    if color[0] == -1:
        plot.plot_surface(xx, yy, z)
    else:
        plot.plot_surface(xx, yy, z,color=color)

def addPoints(plot,array,color=[-1]):
    plot.set_xlim(-30,30)
    plot.set_ylim(-30,30)
    plot.set_zlim(-30,30)

    Tarray = array.T
    if color[0] == -1:
        plot.scatter(Tarray[0],Tarray[1],Tarray[2,])
    else:
        plot.scatter(Tarray[0],Tarray[1],Tarray[2],color=color)

def showPoints(array,color=[-1]):
    fig = plt.figure()
    plot = fig.add_subplot(projection='3d')

    addPoints(plot,array,color)
    
    plt.show()

def showPointsWithGradation(array):
    fig = plt.figure()
    plot = fig.add_subplot(projection='3d')

    for i in range(array.shape[0]):
        addPoints(plot,array[i:i+1],[i/array.shape[0],i/array.shape[0],i/array.shape[0]])
    
    plt.show()

def showPointsGroups(array,color=[-1]):
    fig = plt.figure()
    plot = fig.add_subplot(projection='3d')
    for pts in array:
        addPoints(plot,pts,color)
    
    plt.show()

def showRectangles(rectangles):
    fig = plt.figure()
    plot = fig.add_subplot(projection='3d')
    plot.set_xlim(-30,30)
    plot.set_ylim(-30,30)
    plot.set_zlim(-30,30)
    for rect in rectangles:
        rectangle = Poly3DCollection([rect])
        rectangle.set_facecolor(np.random.rand(3))
        plot.add_collection3d(rectangle)
    plt.show()

def showPointsWithMatlab(eng,points,timeString,step):
     eng.workspace['x'] = double_m(points[:,0].tolist())
     eng.workspace['y'] = double_m(points[:,1].tolist())
     eng.workspace['z'] = double_m(points[:,2].tolist())
     eng.eval('scatter3(x,y,z,"filled")')
     eng.savefig(timeString + "s" + str(step)+ ".fig",nargout=0)

def showPointsGroupsWithMatlab(eng,Groups,timeString,step):
     eng.workspace['x'] = double_m(Groups[0][:,0].tolist())
     eng.workspace['y'] = double_m(Groups[0][:,1].tolist())
     eng.workspace['z'] = double_m(Groups[0][:,2].tolist())
     eng.workspace['s'] = double_m([36] * Groups[0].shape[0])
     eng.workspace['c'] = double_m(random.rand(3))
     eng.eval('scatter3(x,y,z,s,c,"filled")')
     eng.hold('on',nargout=0)

     for i in range(1,len(Groups)):
        eng.workspace['x'] = double_m(Groups[i][:,0].tolist())
        eng.workspace['y'] = double_m(Groups[i][:,1].tolist())
        eng.workspace['z'] = double_m(Groups[i][:,2].tolist())
        eng.workspace['s'] = double_m([36] * Groups[i].shape[0])
        eng.workspace['c'] = double_m(random.rand(3))
        eng.eval('scatter3(x,y,z,s,c,"filled")')

     eng.hold("off",nargout=0)

     eng.savefig(timeString + "s" + str(step)+ ".fig",nargout=0)

def showRectanglessWithMatlab(eng,rectangles,timeString,step):
     eng.workspace['x'] = double_m(rectangles[0][:,0].tolist())
     eng.workspace['y'] = double_m(rectangles[0][:,1].tolist())
     eng.workspace['z'] = double_m(rectangles[0][:,2].tolist())
     eng.workspace['c'] = double_m(random.rand(3))

     eng.eval("fill3(x,y,z,c)")
     eng.hold('on',nargout=0)

     for i in range(1,len(rectangles)):
        eng.workspace['x'] = double_m(rectangles[i][:,0].tolist())
        eng.workspace['y'] = double_m(rectangles[i][:,1].tolist())
        eng.workspace['z'] = double_m(rectangles[i][:,2].tolist())
        eng.workspace['c'] = double_m(random.rand(3))
        eng.eval("fill3(x,y,z,c)")

     eng.hold("off",nargout=0)

     eng.savefig(timeString + "s" + str(step)+ ".fig",nargout=0)
         



