import jax
import jax.numpy as jnp
import os

class Vehicle(object):
    def __init__(self, dynamics_model, action_space_size, state0, n, thruster_positions, thruster_force_vectors, dt=0.01):
        """
        INPUTS:
            dynamics_model: a function that takes in the state and control
                input and returns the state at the next timestep
            state0: 6x1 numpy array of the initial state (x,y,z,xdot,ydot,zdot)
            n: mean motion of the chaser satellite (n=mu**0.5/a**1.5)
            B: 6x3 numpy array of the control input matrix (optional)
        """
        self.dynamics_model = dynamics_model
        self.state = state0
        self.n = n

        self.action_space_size = action_space_size

        self.dt = dt
        self.t = 0

        self.state_trajectory = [jnp.insert(state0,0,0)]
        self.control_trajectory = []

        self.thruster_positions = thruster_positions
        self.thruster_force_vectors = thruster_force_vectors

        #Populate A, B from the dynamics model at (state0, 0)
        self.calculateLinearControlMatrices()

    def calculateLinearControlMatrices(self):
        # jax.jacfwd(self.dynamics_model, argnums=(0,2))(self.state, self.n, jnp.zeros((3,1)))
        #Use JAX to calculate the A, B, and C matrices for the function self.dynamics_model
        self.A = jax.jacfwd(self.dynamics_model, argnums=0)(self.state, jnp.zeros((self.action_space_size)), self.n, self.thruster_positions, self.thruster_force_vectors)
        self.B = jax.jacfwd(self.dynamics_model, argnums=1)(self.state, jnp.zeros((self.action_space_size)), self.n, self.thruster_positions, self.thruster_force_vectors)

        # print(jax.jacfwd(self.dynamics_model, argnums=(0,2))(self.state, self.n, jnp.zeros((3,1))))

    def propagateVehicleState(self, controls):
        """
        Propagate the dynamics of the system forward one timestep according to
        the provided dynamics model.
        """
        self.state = self.dynamics_model(self.state, controls, self.n, self.thruster_positions, self.thruster_force_vectors, dt=self.dt)
        
        self.t += self.dt

        self.state_trajectory.append(jnp.insert(self.state,0,self.t))
        self.control_trajectory.append(jnp.insert(controls,0,self.t))


        return self.state
    
    def saveTrajectoryLog(self, dir_name):
        os.mkdir(dir_name)
        state_traj = dir_name + "/state_traj.csv"
        with open(state_traj, 'w+') as f: #Open the file for writing, or else create it
            print(f"Saving trajectory log to {os.path.abspath(state_traj)}")
            f.write("t,x,y,z,xdot,ydot,zdot,q_scalar,q_x,q_y,q_z,wx,wy,wz\n")
            for state in self.state_trajectory:
                f.write(",".join([str(x) for x in state]) + "\n")

        control_traj = dir_name + "/control_traj.csv"
        with open(control_traj, 'w+') as f:
            num_controls = len(self.control_trajectory[0])-1
            header_str = 't,'+','.join(f"u{i}" for i in range(num_controls))+"\n"
            print(f"Saving control trajectory log to {os.path.abspath(control_traj)}")
            f.write(header_str)
            for control in self.control_trajectory:
                f.write(",".join([str(x) for x in control]) + "\n")