import jax.numpy as jnp
from jaxlie import SO3
import numpy as np

def simpleVehicleDynamics(state, control_input, n, thruster_positions=None, thruster_force_vectors=None, dt=0.01):
    """
    Dynamics model for a simple vehicle with CW dynamics
    and controls which directly impact velocities.

    Accepts thruster positions and force vectors, but ignores them.
    """
    A = jnp.array(([0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1],
                                    [3*n**2, 0, 0, 0, 2*n, 0],
                                    [0, 0, 0, -2*n, 0, 0],
                                    [0, 0, -n**2, 0, 0, 0]))

    B = jnp.array(([0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]))
    
    state = state + (A@state + B@control_input)*dt

    return state

def thrustersVehicleDynamics(state, control_input, n, thruster_positions, thruster_force_vectors, dt=0.01, deterministic=False):
    """
    Dynamics model for a vehicle with an arbitrary amount of thrusters with arbitrary positions and orientations,
    and linear position dynamics obeying the Clohessy-Wiltshire equations.
    """
    ###### CALCULATE THE STATE EVOLUTION, WITHOUT TAKING INTO ACCOUNT THRUSTERS YET ######
    #(6,13) matrix which defines the evolution of the x,y,z, xdot, ydot, zdot state variables over time.
    cartesian_dynamics = jnp.hstack((jnp.array(([0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1],
                                    [3*n**2, 0, 0, 0, 2*n, 0],
                                    [0, 0, 0, -2*n, 0, 0],
                                    [0, 0, -n**2, 0, 0, 0])), jnp.zeros((6,7))))
    
    # #Calculate the initial quaternion
    q_curr = SO3(state[6:10])

    angular_rates = state[10:13]
    angular_rate_quat = SO3(jnp.hstack((0, angular_rates)))

    q_next = SO3.multiply(q_curr, SO3.exp(angular_rates*dt/2))
    
    #Concatenate all the dynamics together
    #States 1-6 (x,y,z,dx,dy,dz) are evolved by the Clohessy Wilthsire equations, multiplied by the timestep
    #States 7-10 (quaternion) are evolved by the discrete time quaternion dynamics
    #States 11-13 (wx,wy,wz) do not evolve over time except for the addition of the control input
    Ax = jnp.hstack((state[0:6]+(cartesian_dynamics@state)*dt, q_next.wxyz, state[10:13]))

    ###### CALCULATE THE THRUSTER FORCES AND TORQUES ######
    thruster_force = 10
    control_input = thruster_force*control_input

    #Define inertia
    I = jnp.array(([1,0,0],
                    [0,1,0],
                    [0,0,1]))
    mass = 1

    #Take rxF to get the torque vector (3x12)
    thruster_torques = jnp.transpose(jnp.cross(thruster_positions, thruster_force_vectors))

    #Multiply by control input to get the torques (3,)
    control_torques = thruster_torques@control_input

    rpy_accels = jnp.linalg.inv(I)@control_torques

    #Calculate the forces in the body frame
    body_frame_forces = jnp.transpose(thruster_force_vectors)@control_input

    #Calculate the accelerations in the inertial frame
    xyz_accels = q_curr.apply(body_frame_forces/mass)

    thruster_force_vectors = jnp.hstack((jnp.eye(3), -jnp.eye(3)))
    thruster_force_vectors = jnp.vstack((jnp.zeros((3,6)), thruster_force_vectors))

    Bu = jnp.hstack((jnp.zeros(3), xyz_accels, jnp.zeros(4), rpy_accels))*dt

    if deterministic:
        noise = np.zeros(13)
    else:
        #only noise on velocities
        noise = np.array([0, 0, 0, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0.01, 0.01, 0.01])

    state = Ax + Bu + noise

    return state

def CWDynamicsAMatrix(n):
    """
    Propagate the dynamics of the system forward one timestep according to
    the Clohessy-Wiltshire equations.

    INPUTS:
        state: 12x1 numpy array of the current state (x,y,z,xdot,ydot,zdot,roll, pitch, yaw, rdot, pdot, ydot)
        n: mean motion of the chaser satellite (n=mu**0.5/a**1.5)
        control_input: 3x1 numpy array of the control input (u,v,w)
        B: 6x3 numpy array of the control input matrix (optional)
        dt: timestep (optional)
    """
    cartesian_dynamics = jnp.hstack((jnp.array(([0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1],
                                    [3*n**2, 0, 0, 0, 2*n, 0],
                                    [0, 0, 0, -2*n, 0, 0],
                                    [0, 0, -n**2, 0, 0, 0])), jnp.zeros((6,7))))
    
    
    attitude_dynamics = jnp.hstack((jnp.zeros((6,6)), 
                                    jnp.array(([0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 0, 1, 0],
                                               [0, 0, 0, 0, 0, 1],
                                               [0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0]))))
    
    A = jnp.vstack((cartesian_dynamics, attitude_dynamics))

    return A