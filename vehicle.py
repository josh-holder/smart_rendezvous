import jax
import jax.numpy as jnp
import os

class Vehicle(object):
    def __init__(self, dynamics_model, state0, n, dt=0.01):
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

        self.dt = dt

        self.state_trajectory = [state0]

        #Populate A, B from the dynamics model at (state0, 0)
        self.calculateLinearControlMatrices()

    def calculateLinearControlMatrices(self):
        # jax.jacfwd(self.dynamics_model, argnums=(0,2))(self.state, self.n, jnp.zeros((3,1)))
        #Use JAX to calculate the A, B, and C matrices for the function self.dynamics_model
        self.A = jax.jacfwd(self.dynamics_model, argnums=0)(self.state, self.n, jnp.zeros((3)))
        self.B = jax.jacfwd(self.dynamics_model, argnums=2)(self.state, self.n, jnp.zeros((3)))

        # print(jax.jacfwd(self.dynamics_model, argnums=(0,2))(self.state, self.n, jnp.zeros((3,1))))

    def propagateVehicleState(self, controls):
        """
        Propagate the dynamics of the system forward one timestep according to
        the provided dynamics model.
        """
        self.state = self.dynamics_model(self.state, self.n, controls, dt=self.dt)
        
        self.state_trajectory.append(self.state)

        return self.state
    
    def saveTrajectoryLog(self, file_name):
        with open(file_name, 'w+') as f: #Open the file for writing, or else create it
            print(f"Saving trajectory log to {os.path.abspath(file_name)}")
            f.write("x,y,z,xdot,ydot,zdot\n")
            t = 0
            for state in self.state_trajectory:
                f.write(",".join([str(t)]+[str(x) for x in state]) + "\n")
                t += self.dt