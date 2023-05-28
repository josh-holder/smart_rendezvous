from vehicle import Vehicle
from dynamics import simpleVehicleDynamics, thrustersVehicleDynamics
from traj_opt import optimize_trajectory

import numpy as np
import jax.numpy as jnp
import jax

from jaxlie import SO3

from traj_plot import plot_trajectory, make_video, readLogFile

from control import dlqr
import argparse
import os

def _get_command_line_args():
    parser = argparse.ArgumentParser(description='Run the simulation')
    parser.add_argument('--dt', type=float, default=0.01, help='Timestep for simulation')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for simulation')
    parser.add_argument('-r','--run-folder', type=str, default=None, help='Path to run files. default is runs/run_{seed}')
    parser.add_argument('--log', action='store_true', help='Flag determining whether or not to save logs of the run.')
    parser.add_argument('--plot', action='store_true', help='Flag determining whether to plot the trajectory')
    parser.add_argument("--video", action='store_true', help='Flag determining whether to make a video of the trajectory')
    parser.add_argument("--tol", type=float, default=0.01, help='Tolerance for final state error')

    args = parser.parse_args()

    #Plotting infrastructure
    if args.seed == None: args.seed = np.random.randint(100000)
    
    if args.run_folder is None:
        args.run_folder = "runs/run_{}".format(args.seed)
    else:
        args.run_folder = "runs/{}".format(args.run_folder)
    
    return args

if __name__ == "__main__":
    args = _get_command_line_args()

    np.random.seed(args.seed)
    jax_rng_key = jax.random.PRNGKey(args.seed)

    #Initialize 12x1 initial state vector (0.9725809, 0.018864, 0.0850898, 0.215616 corresponds to 0, 10, 25 XYZ euler angles)
    initial_state = jnp.array([-10.0,1.0,1.0, 1.0,1.0,1.0, 0.9725809, 0.018864, 0.0850898, 0.215616, 0,0,0])

    mu = 3.816e14 # Gravitational parameter of earth
    #initialize mean motion of earth orbit of 6378km
    n = jnp.sqrt(mu/(6738000**3))

    # Define thruster positions
    thruster_positions = jnp.array(([0.5,0,0],
                                    [0.5,0,0],
                                    [-0.5,0,0],
                                    [-0.5,0,0],
                                    [0,0.5,0],
                                    [0,0.5,0],
                                    [0,-0.5,0],
                                    [0,-0.5,0],
                                    [0,0,0.5],
                                    [0,0,0.5],
                                    [0,0,-0.5],
                                    [0,0,-0.5]))
    
    #Define thruster force vectors, with them pointing 45 degrees from the surface normal.
    thruster_force_vectors = -1*jnp.array(([0.707, 0.707, 0],
                                        [0.707, -0.707, 0],
                                        [-0.707, 0.707, 0],
                                        [-0.707, -0.707, 0],
                                        [0, 0.707, 0.707],
                                        [0, 0.707, -0.707],
                                        [0, -0.707, 0.707],
                                        [0, -0.707, -0.707],
                                        [0.707, 0, 0.707],
                                        [-0.707, 0, 0.707],
                                        [0.707, 0, -0.707],
                                        [-0.707, 0, -0.707]))

    vehicle = Vehicle(thrustersVehicleDynamics, 12, initial_state, n, thruster_positions, thruster_force_vectors, dt=args.dt, deterministic=True)

    desired_state = jnp.array([0,0,0,0,0,0,1,0,0,0,0,0,0], dtype=np.float32)

    optimize_trajectory(vehicle, initial_state, desired_state, dt=args.dt, tolerance=args.tol, verbose=True)

    # make video from existing logs
    # state_data = readLogFile("runs/simple_docking_test/state_traj.csv")
    # control_data = readLogFile("runs/simple_docking_test/control_traj.csv")
    # make_video("runs/simple_docking_test", state_data, control_data, thruster_positions, thruster_force_vectors, args.dt, desired_state=desired_state)

    if args.log: vehicle.saveTrajectoryLog(args.run_folder)
    if args.plot: plot_trajectory(vehicle.state_trajectory, vehicle.control_trajectory, args.dt, args.run_folder)
    if args.video: make_video(args.run_folder, vehicle.state_trajectory, vehicle.control_trajectory, thruster_positions, thruster_force_vectors, args.dt, desired_state=desired_state)