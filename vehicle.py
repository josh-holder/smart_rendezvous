import jax
import jax.numpy as jnp
import os

class Vehicle(object):
    def __init__(self, dynamics_model, action_space_size, state0, n, thruster_positions, thruster_force_vectors, dt=0.01):
        """
        INPUTS:
            dynamics_model: a function that takes in the state and control
                input and returns the state at the next timestep
            state0: 13x1 numpy array of the initial state (x,y,z,xdot,ydot,zdot, quat, wx,wy,wz)
            n: mean motion of the chaser satellite (n=mu**0.5/a**1.5)
            TODO: finish inputs
        """
        self.dynamics_model = dynamics_model
        self.initial_state = state0
        self.state = state0
        self.n = n

        self.action_space_size = action_space_size

        self.dt = dt
        self.t = 0

        self.state_trajectory = [state0]
        self.control_trajectory = []

        self.thruster_positions = thruster_positions
        self.thruster_force_vectors = thruster_force_vectors

        #Populate A, B from the dynamics model at (state0, 0)
        self.A, self.B, self.C = self.calculateLinearControlMatrices(state0, jnp.zeros((self.action_space_size)))

    def calculateLinearControlMatrices(self, state, control):
        # jax.jacfwd(self.dynamics_model, argnums=(0,2))(self.state, self.n, jnp.zeros((3,1)))
        #Use JAX to calculate the A, B, and C matrices for the function self.dynamics_model
        A = jax.jacfwd(self.dynamics_model, argnums=0)(state, control, self.n, self.thruster_positions, self.thruster_force_vectors)
        B = jax.jacfwd(self.dynamics_model, argnums=1)(state, control, self.n, self.thruster_positions, self.thruster_force_vectors)

        C = self.dynamics_model(state, control, self.n, self.thruster_positions, self.thruster_force_vectors, dt=self.dt) - A@state - B@control 

        return A, B, C

    def propagateVehicleState(self, controls):
        """
        Propagate the dynamics of the system forward one timestep according to
        the provided dynamics model.
        """
        self.state = self.dynamics_model(self.state, controls, self.n, self.thruster_positions, self.thruster_force_vectors, dt=self.dt)
        
        self.t += self.dt

        self.state_trajectory.append(self.state)
        self.control_trajectory.append(controls)


        return self.state

    def resetVehicle(self):
        """
        Reset vehicle to the initial state and time, clearning the trajectory log
        """
        self.state = self.initial_state

        self.t = 0

        self.state_trajectory = [self.initial_state]
        self.control_trajectory = []

    def propogateVehicleTrajectory(self, control_inputs):
        """
        Given a sequence of control inputs, propagate the vehicle forward in time
        control_inputs is a (horizon_length x num_controls) numpy array
        """
        for control in control_inputs:
            self.propagateVehicleState(control)
    
    def saveTrajectoryLog(self, dir_name):
        if not os.path.exists(dir_name): os.mkdir(dir_name)

        state_traj = dir_name + "/state_traj.csv"
        with open(state_traj, 'w+') as f: #Open the file for writing, or else create it
            print(f"Saving trajectory log to {os.path.abspath(state_traj)}")
            f.write("t,x,y,z,xdot,ydot,zdot,q_scalar,q_x,q_y,q_z,wx,wy,wz\n")
            t = 0
            for state in self.state_trajectory:
                f.write(",".join([t]+[str(x) for x in state]) + "\n")
                t += self.dt

        control_traj = dir_name + "/control_traj.csv"
        with open(control_traj, 'w+') as f:
            num_controls = len(self.control_trajectory[0])-1
            header_str = 't,'+','.join(f"u{i}" for i in range(num_controls))+"\n"
            print(f"Saving control trajectory log to {os.path.abspath(control_traj)}")
            f.write(header_str)
            t = 0
            for control in self.control_trajectory:
                f.write(",".join([t] + [str(x) for x in control]) + "\n")
                t += self.dt