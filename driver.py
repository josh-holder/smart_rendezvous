from vehicle import Vehicle
from cw_dynamics import propagateCWDynamics
import numpy as np

from traj_plot import plot_trajectory

from control import dlqr
import argparse
import os

def _get_command_line_args():
    parser = argparse.ArgumentParser(description='Run the simulation')
    parser.add_argument('--dt', type=float, default=0.01, help='Timestep for simulation')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for simulation')
    parser.add_argument('--run-path', type=str, default=None, help='Path to run files. default is runs/run_{seed}')
    parser.add_argument('--log', type=str, default=None, help='Log file to save trajectory to. default is runs/{run_name}/traj_data.csv')
    parser.add_argument('--plot', action='store_true', help='Plot the trajectory')
    parser.add_argument('--plot_name', type=str, default=None, help='Name of the plot file. default is runs/{run_name}/traj_plot.png')

    args = parser.parse_args()

    #Plotting infrastructure
    if args.seed == None: args.seed = np.random.randint(100000)
    np.random.seed(args.seed)
    if args.run_path is None:
        args.run_path = "runs/run_{}".format(args.seed)
        os.mkdir(args.run_path)
    if args.log is None:
        args.log = "{}/traj_data.csv".format(args.run_path)
    if args.plot and args.plot_name is None:
        args.plot_name = "{}/traj_plot.png".format(args.run_path)
    
    return args

if __name__ == "__main__":
    args = _get_command_line_args()

    #Intialize 6x1 initial state vector of all zeros
    initial_state = np.zeros((6))
    initial_state[0] = 10
    initial_state[1] = 5
    initial_state[2] = -3

    mu = 3.816e14 # Gravitational parameter of earth
    #initialize mean motion of earth orbit of 6378km
    n = np.sqrt(mu/(6738000**3))

    vehicle = Vehicle(propagateCWDynamics, initial_state, n, dt=args.dt)

    controls = np.array(([1,0,0]))

    #Compute the LQR gain matrix:
    K, S, E = dlqr(vehicle.A, vehicle.B, np.eye(6), np.eye(3))

    #loop until norm of the state vector is less than .001:
    while np.linalg.norm(vehicle.state) > 0.001:
        #Propagate the vehicle state
        vehicle.propagateVehicleState(-K@vehicle.state)
    
    print(args.log)
    vehicle.saveTrajectoryLog(args.log)
    
    if args.plot: plot_trajectory(args.log ,args.plot_name)
