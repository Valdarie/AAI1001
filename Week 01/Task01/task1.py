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


'''
In this updated code, the createacovariance() function generates a random symmetric matrix S and computes its eigendecomposition. It then scales the eigenvalues of the matrix to obtain a covariance matrix cov. The plotellipse() function uses the covariance matrix to generate num_points points on the surface of a sphere, and then scales the points using the Cholesky decomposition of the covariance matrix to obtain the ellipse. Finally, it plots the scaled points on a 3D plot using the plot_surface() function from matplotlib.

To test the code, you can simply call plotellipse() from the main function. The resulting plot should show a 3D ellipse with its principal axes aligned with the eigenvectors of the covariance matrix.
'''