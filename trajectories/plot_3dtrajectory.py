import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import h5py
import sys
from lib.constants import *


def plot(fig, trajectory, sensor_height):
    # Filter out everything above block (start of trajectory)
    traj_inside = trajectory[trajectory[:, 2] < sensor_height]

    ax = fig.add_subplot(111, projection='3d')  # type:Axes3D
    ax.view_init(elev=20., azim=32)

    # Plot track
    ax.plot(traj_inside[:, 0], traj_inside[:, 1], traj_inside[:, 2])
    # Plot incident electron
    print("Traj incident: %f, %f" % (traj_inside[0, 1] / 55000, traj_inside[0, 0] / 55000))
    ax.plot([traj_inside[0, 0]], [traj_inside[0, 1]], [sensor_height], 'ro')

    # Set axes scale
    ax.set_xlim3d(0, n_pixels * pixel_size)
    ax.set_ylim3d(0, n_pixels * pixel_size)
    ax.set_zlim3d(0, sensor_height)

    # Set tick lines to pixel size,
    xedges = np.arange(0, n_pixels * pixel_size, pixel_size)
    yedges = np.arange(0, n_pixels * pixel_size, pixel_size)
    zedges = np.arange(0, sensor_height, 100000)
    ax.set_yticks(yedges, minor=False)
    ax.yaxis.grid(True, which='major')
    ax.yaxis.set_ticklabels([])
    ax.set_xticks(xedges, minor=False)
    ax.xaxis.grid(True, which='major')
    ax.xaxis.set_ticklabels([])
    ax.set_zticks(zedges, minor=False)
    ax.zaxis.grid(True, which='major')

    # Change background color
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    return fig, ax


if __name__ == "__main__":
    filename = sys.argv[1]

    f = h5py.File(filename, 'r')

    # Get trajectory
    trajectory = f['trajectories'][sys.argv[2]][()]
    height = f.attrs['sensor_height']

    ax = plot(plt.figure(), trajectory, height)

    # from matplotlib import animation
    # def animate(i):
    #     ax[1].view_init(30, i)
    #     plt.draw()
    #     plt.pause(.001)
    # # Animate
    # anim = animation.FuncAnimation(ax[0], animate,
    #                                frames=360, interval=20)
    # # Save
    # anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()
