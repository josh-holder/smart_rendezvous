import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxlie import SO3
from matplotlib.animation import FuncAnimation
import os
from matplotlib.patches import Polygon

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

def plot_trajectory(state_traj, control_traj, dt, run_folder_name):
    """
    Given a file name correpsonding to a csv with the following format:
    t, x, y, z, xdot, ydot, zdot

    Plot the trajectory of the vehicle in each state over time, in 6 subplots
    """
    if not os.path.exists(run_folder_name): os.mkdir(run_folder_name)
    
    data = jnp.vstack(state_traj)
    t = np.linspace(0,dt*data.shape[0],data.shape[0])
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    xdot = data[:,3]
    ydot = data[:,4]
    zdot = data[:,5]

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

    traj_plot_name = run_folder_name+"/trajectory.png"
    plt.savefig(traj_plot_name)

    print(f"Saving plot to {traj_plot_name}")

def make_video(run_folder_name, state_traj, control_traj, base_thruster_positions, base_thruster_vectors, dt, expected_state_traj=None, desired_state=None):
    if not os.path.exists(run_folder_name): os.mkdir(run_folder_name)
    
    state_data = jnp.vstack(state_traj) #stack the list of states to an array of states

    control_data = jnp.vstack(control_traj)
    max_control_input = control_data.max()
    control_data = control_data/max_control_input #normalize the control inputs

    #angular stuff
    fig = plt.figure()

    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    def update(frame):
        print(f"Creating video: frame {frame}/{state_data.shape[0]}",end='\r')
        ax.clear()
        ax.set_xlim(-8,8)
        ax.set_ylim(-3,3)
        ax.set_zlim(-6,6)
        ax.set_aspect('equal')

        #Plot the heading of the spacecraft
        xyz = state_data[frame,0:3]
        quat = SO3(state_data[frame,6:10]).normalize()
        base_vector = np.array([5.0,0.0,0.0])
        new_vect = quat.apply(base_vector)
        quiv = ax.quiver(*xyz,*new_vect)

        #Plot the body of the spacecraft
        faces = create_box(xyz, quat)
        face_colors = ['red', 'green', 'blue', 'red', 'green', 'blue']
        for face, face_color in zip(faces, face_colors):
            ax.add_collection3d(Poly3DCollection([face], facecolors=face_color, linewidths=1, alpha=1))

        #Plot the active thruster directions
        thruster_vectors, thruster_positions = create_thrusters(xyz, quat, control_data[frame,:], base_thruster_positions, base_thruster_vectors)

        for thruster_num in range(thruster_vectors.shape[0]):
            thruster_vector = thruster_vectors[thruster_num,:]
            thruster_pos = thruster_positions[thruster_num,:]
            ax.quiver(*(xyz+thruster_pos), *thruster_vector,color='green')

        #Plot goal position
        if desired_state is not None:
            xyz = desired_state[0:3]
            quat = SO3(desired_state[6:10]).normalize()
            base_vector = np.array([5.0,0.0,0.0])
            new_vect = quat.apply(base_vector)
            quiv = ax.quiver(*xyz,*new_vect, color='green', alpha=0.5)

            #Plot the body of the spacecraft
            faces = create_box(xyz, quat)
            for face in faces:
                ax.add_collection3d(Poly3DCollection([face], facecolors='green', linewidths=1, alpha=0.5))

        
        #If there is an expected state trajectory, plot it
        if expected_state_traj is not None:
            #Plot the heading of the spacecraft
            xyz = expected_state_traj[frame,0:3]
            quat = SO3(expected_state_traj[frame,6:10]).normalize()
            base_vector = np.array([5.0, 0.0, 0.0])
            new_vect = quat.apply(base_vector)
            quiv = ax.quiver(*xyz,*new_vect, alpha=0.25)

            #Plot the body of the spacecraft
            faces = create_box(xyz, quat)
            face_colors = ['red', 'green', 'blue', 'red', 'green', 'blue']
            for face, face_color in zip(faces, face_colors):
                ax.add_collection3d(Poly3DCollection([face], facecolors=face_color, linewidths=1, alpha=0.25))

    ani = FuncAnimation(fig, update, frames=state_data.shape[0], interval=dt*1500) #dt in milliseconds

    animation_name = run_folder_name+'/animation.mp4'
    ani.save(animation_name, writer='ffmpeg')

    print(f"Saving plot to {animation_name}")

    plt.show()

def create_box(xyz, quat):
    """
    Creates a box with 0.5 side length at the specified position
    """
    points = jnp.array([[0.5, 0.5, 0.5],
                        [0.5, 0.5, -0.5],
                        [0.5, -0.5, 0.5],
                        [0.5, -0.5, -0.5],
                        [-0.5, 0.5, 0.5],
                        [-0.5, 0.5, -0.5],
                        [-0.5, -0.5, 0.5],
                        [-0.5, -0.5, -0.5]])
    
    px_face_indices = [0,1,3,2] #all the points with a positive x coordinate are on a single face
    nx_face_indices = [4,5,7,6]

    py_face_indices = [0,1,5,4]
    ny_face_indices = [2,3,7,6]

    pz_face_indices = [0,2,6,4]
    nz_face_indices = [1,3,7,5]

    #Rotate the points by the quaternion
    points = jnp.apply_along_axis(quat.apply, axis=1, arr=points) 

    points += xyz #add translation

    px_face = [points[i,:] for i in px_face_indices]
    py_face = [points[i,:] for i in py_face_indices]
    pz_face = [points[i,:] for i in pz_face_indices]

    nx_face = [points[i,:] for i in nx_face_indices]
    ny_face = [points[i,:] for i in ny_face_indices]
    nz_face = [points[i,:] for i in nz_face_indices]

    return [px_face, py_face, pz_face, nx_face, ny_face, nz_face]

def create_thrusters(xyz, quat, controls, thruster_positions, thruster_vectors):
    """
    Creates an arrow pointing in the thruster firing direction
    """
    #Determine if we should consider a thruster as on
    # (currently, if it's larger than the mean)
    mean_control = controls.mean()
    controls_on = jnp.where(controls>mean_control+0.001, 1, 0)

    #Rotate the thruster vectors by the quaternion
    
    # scalings = jnp.ones_like(controls)
    # scalings = [1 if ctrl > mean_control else 0 for ctrl in controls]
    thruster_vectors = jnp.apply_along_axis(quat.apply, axis=1, arr=thruster_vectors) 
    #Scale the thruster vectors by the control input
    # thruster_vectors = jnp.apply_along_axis(controls_on, axis=1, arr=thruster_vectors)
    thruster_vectors = controls_on[:,jnp.newaxis] * (-1*thruster_vectors) #reverse for visualizing thruster firing
    # thruster_vectors = jnp.where(controls > mean_control,(-1*thruster_vectors), jnp.zeros_like(thruster_vectors)) 

    thruster_positions = jnp.apply_along_axis(quat.apply, axis=1, arr=thruster_positions) 

    return thruster_vectors, thruster_positions

def readLogFile(log_file):
    """
    Given log file from a run of this algorithm, loads data into lists for use in other plotting algorithms
    """

    data = np.loadtxt(log_file, delimiter=',',skiprows=1)

    return data[:,1:] #return the data without the time index