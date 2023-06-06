import gym
from gym import Env, logger, spaces, utils
import jax.numpy as jnp
from jaxlie import SO3
import numpy as np

class DockingGym(Env):
    metadata = {'render.modes': ['human','rgb_array']}

    def __init__(self, dynamics_model, action_space_size, state0, n, thruster_positions, thruster_force_vectors, dt=0.01, deterministic=False):
        self.initial_state = state0
        self.state = state0
        self.n = n

        self.action_space_size = action_space_size

        self.dt = dt
        self.t = 0

        self.deterministic = deterministic

        self.state_trajectory = [state0]
        self.control_trajectory = []

        self.thruster_positions = thruster_positions
        self.thruster_force_vectors = thruster_force_vectors

    def step(self, action):
        cartesian_dynamics = jnp.hstack((jnp.array(([0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1],
                                    [3*self.n**2, 0, 0, 0, 2*self.n, 0],
                                    [0, 0, 0, -2*self.n, 0, 0],
                                    [0, 0, -self.n**2, 0, 0, 0])), jnp.zeros((6,7))))
    
        # #Calculate the initial quaternion
        q_curr = SO3(state[6:10])

        angular_rates = state[10:13]
        angular_rate_quat = SO3(jnp.hstack((0, angular_rates)))

        q_next = SO3.multiply(q_curr, SO3.exp(angular_rates*self.dt/2))
        
        #Concatenate all the dynamics together
        #States 1-6 (x,y,z,dx,dy,dz) are evolved by the Clohessy Wilthsire equations, multiplied by the timestep
        #States 7-10 (quaternion) are evolved by the discrete time quaternion dynamics
        #States 11-13 (wx,wy,wz) do not evolve over time except for the addition of the control input
        Ax = jnp.hstack((state[0:6]+(cartesian_dynamics@state)*self.dt, q_next.wxyz, state[10:13]))

        ###### CALCULATE THE THRUSTER FORCES AND TORQUES ######
        thruster_force = 10
        control_input = thruster_force*control_input

        #Define inertia
        I = jnp.array(([1,0,0],
                        [0,1,0],
                        [0,0,1]))
        mass = 1

        #Take rxF to get the torque vector (3x12)
        thruster_torques = jnp.transpose(jnp.cross(self.thruster_positions, thruster_force_vectors))

        #Multiply by control input to get the torques (3,)
        control_torques = thruster_torques@control_input

        rpy_accels = jnp.linalg.inv(I)@control_torques

        #Calculate the forces in the body frame
        body_frame_forces = jnp.transpose(thruster_force_vectors)@control_input

        #Calculate the accelerations in the inertial frame
        xyz_accels = q_curr.apply(body_frame_forces/mass)

        thruster_force_vectors = jnp.hstack((jnp.eye(3), -jnp.eye(3)))
        thruster_force_vectors = jnp.vstack((jnp.zeros((3,6)), thruster_force_vectors))

        Bu = jnp.hstack((jnp.zeros(3), xyz_accels, jnp.zeros(4), rpy_accels))*self.dt

        if self.deterministic:
            noise = np.zeros(13)
        else:
            #only noise on velocities
            noise = np.array([0, 0, 0, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0.01, 0.01, 0.01])

        state = Ax + Bu + noise

        return state

    def reset(self):
        observation = self.env.reset()
        return observation

    def render(self, mode='human'):
        self.env.render(mode=mode)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed=seed)

    def configure(self, *args, **kwargs):
        self.env.configure(*args, **kwargs)

    def __str__(self):
        return self.env.__str__()

    def __repr__(self):
        return self.env.__repr__()

    def __getattr__(self, name):
        return getattr(self.env, name)