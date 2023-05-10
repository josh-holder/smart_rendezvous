import numpy as np

def propagateCWDynamics(state, n, control_input, dt=0.01):
    """
    Propagate the dynamics of the system forward one timestep according to
    the Clohessy-Wiltshire equations.

    INPUTS:
        state: 6x1 numpy array of the current state (x,y,z,xdot,ydot,zdot)
        n: mean motion of the chaser satellite (n=mu**0.5/a**1.5)
        control_input: 3x1 numpy array of the control input (u,v,w)
        B: 6x3 numpy array of the control input matrix (optional)
        dt: timestep (optional)
    """
    A = np.array(([0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [3*n**2, 0, 0, 0, 2*n, 0],
                [0, 0, 0, -2*n, 0, 0],
                [0, 0, -n**2, 0, 0, 0]))
        
    B = np.array(([0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]))

    state = state + (A@state + B@control_input)*dt

    return state