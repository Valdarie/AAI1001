import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def createacovariance():

  rng = np.random.default_rng()
  a = rng.random(size=(3, 3))

  s1 = np.matmul(a.T, a)

  w, v = np.linalg.eig(s1)

  # you can play with the non-zero values
  d = np.array([[4, 0, 0], [0, 1, 0], [0, 0, 0.25]])

  a = v.T @ d @ v
  print(a)
  return a


def plotdirvariances(S):

  t = np.linspace(0, 2*np.pi, 50)
  u = np.linspace(0, 2*np.pi, 50)
  T, U = np.meshgrid(t, u)

  V = np.array([np.cos(U) * np.sin(T), np.cos(U) * np.cos(T), np.sin(U)])

  print(V.shape)  # (3,50,50)

  scaledv = np.zeros_like(V)
  for i in range(V.shape[1]):
    for j in range(V.shape[2]):
      thisv = V[:, i, j]

      # TODO need to compute v^T S v
      stv = np.dot(thisv.T, S)
      #TODO multiply it to it
      scaledv[:, i, j] = np.dot(stv, thisv) *thisv

      #scaledv[:,i,j]= stv * V[:,i,j] 
      
      
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  #surf = ax.plot_surface(V[0,:], V[1,:], V[2,:] , antialiased=False, color=(0.9,0.9,0.0,0.2))

  surf = ax.plot_surface(scaledv[0,:,:], scaledv[1,:,:], scaledv[2,:,:],
                       antialiased=False, color=(0.2,0.9,0.0,0.2))

  plt.show()
  
if __name__=='__main__':
  S = createacovariance()
  plotdirvariances(S)

  
