from vehicle import Vehicle
from dynamics import simpleVehicleDynamics, thrustersVehicleDynamics
from traj_opt import optimize_trajectory

import numpy as np
import jax.numpy as jnp
import jax

from jaxlie import SO3

from traj_plot import plot_trajectory, make_video

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

    #Initialize 12x1 initial state vector (0.4330127, 0.0794593, 0.4330127, 0.7865661 corresponds to 45, 30, 45 XYZ euler angles)
    initial_state = jnp.array([-1.0,1.0,1.0, 0,1.0,0, 0.4330127, 0.0794593, 0.4330127, 0.7865661, 0,0,0])

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

    vehicle = Vehicle(thrustersVehicleDynamics, 12, initial_state, n, thruster_positions, thruster_force_vectors, dt=args.dt)

    optimize_trajectory(vehicle, initial_state, jnp.array([0,0,0,0,0,0,1,0,0,0,0,0,0], dtype=np.float32), dt=args.dt)

    #initialize (12,1) control vector
    # control_choices = np.array([0,5,10])
    # controls = np.random.choice(control_choices, size=(12,))
    # controls = np.array([10,0,0,0,0,0,0,0,0,0,0,0])
    # state = initial_state
    # for i in range(100):
    #     vehicle.propagateVehicleState(controls)

    # controls = np.random.choice(control_choices, size=(12,))
    # for i in range(100):
    #     vehicle.propagateVehicleState(controls)

    # print(vehicle.state_trajectory.shape)
    # quats = vehicle.state_trajectory[:,7:11]

    # rpy = quats.as_rpy_radians()
    # print(quats)

    if args.log: vehicle.saveTrajectoryLog(args.run_folder)
    if args.plot: plot_trajectory(vehicle.state_trajectory, vehicle.control_trajectory, args.dt, args.run_folder)
    if args.video: make_video(args.run_folder, vehicle.state_trajectory, vehicle.control_trajectory, thruster_positions, thruster_force_vectors, args.dt)

    # #### COMPUTE FOR SIMPLE SYSTEM #######
    # initial_state = jnp.array([10.0, -5.0, 3.0, 0, 0, 0])

    # vehicle = Vehicle(simpleVehicleDynamics, 3, initial_state, n, thruster_positions, thruster_force_vectors, dt=args.dt)

    # #Compute the LQR gain matrix:
    # K, S, E = dlqr(vehicle.A, vehicle.B, np.eye(6), np.eye(3))

    # #loop until norm of the state vector is less than .001:
    # while np.linalg.norm(vehicle.state) > 0.001:
    #     #Propagate the vehicle state
    #     vehicle.propagateVehicleState(-K@vehicle.state)
    
    # print(args.log)
    # vehicle.saveTrajectoryLog(args.log)
    
    # if args.plot: plot_trajectory(args.log ,args.plot_name)
