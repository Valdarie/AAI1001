import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def createacovariance():

  rng = np.random.default_rng()
  a= rng.random(size=(3,3))
  
  s1=np.matmul( a.T,a )
  
  w,v = np.linalg.eig(s1)
  
  d= np.array ([ [4,0,0], [0,1,0], [0,0,0.25]  ])
  
  a = v.T @ d @ v
  print(a)
  return a


#from https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plotellipse():

  
  t = np.linspace(0, 2*np.pi, 50)
  u = np.linspace(0, 2*np.pi, 50)
  T, U = np.meshgrid(t, u)
  
  V= np.array( [np.cos(U) * np.sin(T), np.cos(U) * np.cos(T), np.sin(U)] )

  
  print(V.shape) #(3,50,50)
  
  scaledv = np.zeros_like(V)
  scalers=np.array([1.0, 4.0, 9.0])
  
  for i in range(V.shape[1]):
    for j in range(V.shape[2]):
      
      scaledv[:,i,j]= V[:,i,j]*scalers
  
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  #surf = ax.plot_surface(V[0,:], V[1,:], V[2,:], antialiased=False, color=(0.9,0.9,0.0,0.2))

  surf = ax.plot_surface(scaledv[0,:,:], scaledv[1,:,:], scaledv[2,:,:],
                       antialiased=False, color=(0.9,0.9,0.0,0.2))

  set_axes_equal(ax) #optional
  plt.show()  

  
if __name__=='__main__':

  plotellipse()
  #plotcircle2()
  
