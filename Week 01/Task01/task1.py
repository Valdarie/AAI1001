import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def createacovariance():
    rng = np.random.default_rng()
    A = rng.random(size=(3,3))
    S = np.matmul(A.T, A)
    w, v = np.linalg.eig(S)
    D = np.array([[4, 0, 0], [0, 1, 0], [0, 0, 0.25]])
    cov = v.T @ D @ v
    return cov

def plotellipse():
    cov = createacovariance()
    mean = np.zeros(3)
    num_points = 50
    t = np.linspace(0, 2*np.pi, num_points)
    u = np.linspace(0, np.pi, num_points)
    T, U = np.meshgrid(t, u)
    X = np.cos(T) * np.sin(U)
    Y = np.sin(T) * np.sin(U)
    Z = np.cos(U)
    XYZ = np.array([X.flatten(), Y.flatten(), Z.flatten()])
    scaled_XYZ = np.linalg.cholesky(cov) @ XYZ
    scaled_X = np.reshape(scaled_XYZ[0,:], (num_points, num_points)) + mean[0]
    scaled_Y = np.reshape(scaled_XYZ[1,:], (num_points, num_points)) + mean[1]
    scaled_Z = np.reshape(scaled_XYZ[2,:], (num_points, num_points)) + mean[2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(scaled_X, scaled_Y, scaled_Z, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

if __name__=='__main__':
    plotellipse()
