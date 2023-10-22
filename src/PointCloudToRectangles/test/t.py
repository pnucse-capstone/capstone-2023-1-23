import numpy as np
import algorithmLib as al
import plotLib as pl


a = np.array([ [[1,0,0],[1,1,0],[0,1,0],[0,0,0]] , [[1.9,0,0],[2.9,0,0.1],[2.9,1,0.1],[1.9,1,0]] , [[0,0,0],[1,0,0],[1,0,1],[0,0,1]]])
pl.showRectangles(a)
a = al.mergeRectangles(a,al.rectanglesToPlaneEquations(a),1,5)
print(a)


pl.showRectangles(a)
