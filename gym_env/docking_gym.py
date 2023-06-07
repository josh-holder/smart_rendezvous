import gym
from gym import Env, logger, spaces, utils
from gym.spaces import MultiBinary
from jaxlie import SO3
import numpy as np

class DockingGym(gym.Env):
    metadata = {'render.modes': ['human','rgb_array']}

    def __init__(self, state0, desired_state, n, \
                 thruster_positions, thruster_force_vectors, dt=0.01, deterministic=False, tol=0.1):
        super(DockingGym, self).__init__()
        
        self.initial_state = state0
        self.state = state0
        self.desired_state = desired_state
        
        self.n = n

        # self.action_space = MultiBinary(np.shape(thruster_positions)[0])
        self.action_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)
        print(self.action_space.sample())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=np.shape(state0), dtype=np.float32)

        self.dt = dt
        self.t = 0
        self.tol = tol

        self.deterministic = deterministic

        self.thruster_positions = thruster_positions
        self.thruster_force_vectors = thruster_force_vectors

    def step(self, action):
        #Take only actions 
        # action = np.where(action>0.5, 1, 0)

        cartesian_dynamics = np.hstack((np.array(([0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1],
                                    [3*self.n**2, 0, 0, 0, 2*self.n, 0],
                                    [0, 0, 0, -2*self.n, 0, 0],
                                    [0, 0, -self.n**2, 0, 0, 0])), np.zeros((6,7))))
    
        # #Calculate the initial quaternion
        q_curr = SO3(self.state[6:10])

        angular_rates = self.state[10:13]
        angular_rate_quat = SO3(np.hstack((0, angular_rates)))

        q_next = SO3.multiply(q_curr, SO3.exp(angular_rates*self.dt/2))
        
        #Concatenate all the dynamics together
        #States 1-6 (x,y,z,dx,dy,dz) are evolved by the Clohessy Wilthsire equations, multiplied by the timestep
        #States 7-10 (quaternion) are evolved by the discrete time quaternion dynamics
        #States 11-13 (wx,wy,wz) do not evolve over time except for the addition of the control input
        Ax = np.hstack((self.state[0:6]+(cartesian_dynamics@self.state)*self.dt, q_next.wxyz, self.state[10:13]))

        ###### CALCULATE THE THRUSTER FORCES AND TORQUES ######
        thruster_force = 10
        control_input = thruster_force*action

        #Define inertia
        I = np.array(([1,0,0],
                        [0,1,0],
                        [0,0,1]))
        mass = 1

        #Take rxF to get the torque vector (3x12)
        thruster_torques = np.transpose(np.cross(self.thruster_positions, self.thruster_force_vectors))

        #Multiply by control input to get the torques (3,)
        control_torques = thruster_torques@control_input

        rpy_accels = np.linalg.inv(I)@control_torques

        #Calculate the forces in the body frame
        body_frame_forces = np.transpose(self.thruster_force_vectors)@control_input

        #Calculate the accelerations in the inertial frame
        xyz_accels = q_curr.apply(body_frame_forces/mass)

        # thruster_force_vectors = np.hstack((np.eye(3), -np.eye(3)))
        # thruster_force_vectors = np.vstack((np.zeros((3,6)), self.thruster_force_vectors))

        Bu = np.hstack((np.zeros(3), xyz_accels, np.zeros(4), rpy_accels))*self.dt

        if self.deterministic:
            noise = np.zeros(13)
        else:
            #only noise on velocities
            noise_std = np.array([0, 0, 0, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0.01, 0.01, 0.01])
            noise = np.random.normal(0, noise_std, 13)

        next_state = Ax + Bu + noise

        reward = self.reward_model(self.state, next_state, action)

        done = False
        if np.linalg.norm(next_state-self.desired_state) < self.tol:
            done = True
            reward += 100

        self.state = next_state
        return next_state, reward, done, {}

    def reset(self, seed=None, options=None):
        self.state = self.initial_state

        return self.state

    def reward_model(self, curr_state, next_state, control):
        """
        Reward model for the docking problem
        """
        curr_state_error = curr_state - self.desired_state
        next_state_error = next_state - self.desired_state

        #Initialize the cost function
        state_importances = np.diag([1,1,1,0.5,0.5,0.5,10,10,10,10,1,1,1])

        weighted_curr_state_error = curr_state_error@state_importances@curr_state_error
        weighted_next_state_error = next_state_error@state_importances@next_state_error

        #If the current error is larger than the next error, then we are getting closer to the desired state
        #and we should be rewarded
        state_diff_reward = weighted_curr_state_error - weighted_next_state_error

        state_reward = next_state_error@state_importances@next_state_error

        print(state_reward, state_diff_reward)

        control_error_reward = -control.sum()/2    

        return (state_diff_reward + control_error_reward)*0.1