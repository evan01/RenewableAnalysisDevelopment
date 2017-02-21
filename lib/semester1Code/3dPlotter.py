from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

def main1():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)
    print("This is actually a pretty cool theme")
    plt.show()

def tutorial():
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    xpos = [-9, 0, 3, 4, 5, 6, 7, 8, 9, 10]
    ypos = [6,7,3,4,5,6,7,8,9,10]
    dz = [1, 2, 3, 4, 20, 10, 7, 8, 9, 10]

    zpos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    num_elements = len(xpos)

    dx = [0,1,1,1,1,1,1,1,1,1]
    dy = [1,1,1,1,1,1,1,1,1,1]


    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')
    plt.show()

if __name__ == '__main__':
    tutorial()