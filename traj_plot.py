import numpy as np
import matplotlib.pyplot as plt
def plot_trajectory(file_name, plot_name='traj_plot.png'):
    """
    Given a file name correpsonding to a csv with the following format:
    t, x, y, z, xdot, ydot, zdot

    Plot the trajectory of the vehicle in each state over time, in 6 subplots
    """
    data = np.loadtxt(file_name, delimiter=',',skiprows=1)
    t = data[:,0]
    x = data[:,1]
    y = data[:,2]
    z = data[:,3]
    xdot = data[:,4]
    ydot = data[:,5]
    zdot = data[:,6]

    plt.figure()
    plt.subplot(3,2,1)
    plt.plot(t,x)
    plt.xlabel('Time (s)')
    plt.ylabel('x (m)')
    plt.subplot(3,2,2)
    plt.plot(t,xdot)
    plt.xlabel('Time (s)')
    plt.ylabel('xdot (m/s)')

    plt.subplot(3,2,3)
    plt.plot(t,y)
    plt.xlabel('Time (s)')
    plt.ylabel('y (m)')
    plt.subplot(3,2,4)
    plt.plot(t,ydot)
    plt.xlabel('Time (s)')
    plt.ylabel('ydot (m/s)')

    plt.subplot(3,2,5)
    plt.plot(t,z)
    plt.xlabel('Time (s)')
    plt.ylabel('z (m)')
    plt.subplot(3,2,6)
    plt.plot(t,zdot)
    plt.xlabel('Time (s)')
    plt.ylabel('zdot (m/s)')
    plt.tight_layout()

    plt.savefig(plot_name)

    print(f"Saving plot to {plot_name}")