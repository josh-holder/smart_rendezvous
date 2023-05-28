import numpy as np
import jax.numpy as jnp

from traj_plot import make_video

import cvxpy as cp
import copy
import time

def initializeOptimizationProblem(time_horizon, As, Bs, Cs, initial_state, desired_state, state_size, control_size):
    """
    Initialize the constraints for the optimization problem
    """
    #Define optimization variables - solving for x and u
    opt_states = cp.Variable((time_horizon+1, state_size))
    opt_controls = cp.Variable((time_horizon, control_size),boolean=True)

    #inequality constraints
    max_control_input = 1
    min_control_input = 0

    equality_constraints = []
    inequality_constraints = []
    for i in range(time_horizon):
        #Initialize the constraints which ensure that the linearized dynamics are satisfied.
        equality_constraints.append(opt_states[i+1,:] == As[i]@opt_states[i,:] + Bs[i]@opt_controls[i,:] + Cs[i])

        #Iniatilize control constraints
        # inequality_constraints.append(opt_controls[i,:] <= max_control_input)
        # inequality_constraints.append(opt_controls[i,:] >= min_control_input)

    #initialize the initial condition constraint
    equality_constraints.append(opt_states[0,:] == initial_state)

    all_constraints = equality_constraints + inequality_constraints

    #Initialize the cost function
    state_importances = np.array([1,1,1,0.5,0.5,0.5,10,10,10,10,1,1,1])
    # state_importances = np.array([1,1,1,0.5,0.5,0.5,0,0,0,0,0,0,0])
    state_costs = np.diag(state_importances) #np.eye(13)
    control_costs = np.eye(12)

    state_cost_per_timestep = [(1/2)*cp.quad_form(opt_states[i,:] - desired_state, state_costs) for i in range(1,time_horizon)]

    control_cost_per_timestep = [(1/2)*cp.quad_form(opt_controls[i,:], control_costs) for i in range(time_horizon)]

    #Final cost is 10x as expensive
    final_cost = (1/2)*cp.quad_form(opt_states[time_horizon,:] - desired_state, 10*state_costs)

    total_cost = cp.sum(state_cost_per_timestep) + cp.sum(control_cost_per_timestep) + final_cost

    prob = cp.Problem(cp.Minimize(total_cost), all_constraints)

    return prob, opt_states, opt_controls

def find_optimal_action(vehicle, initial_state, desired_state, controls_guess, time_horizon, dt=0.01, tolerance=0.01, max_iter=1, verbose=False):
    """
    1. Roll out a trajectory, T_curr, using the dynamics model and random actions
    While T_curr does not end in the desired state, within some tolerance:
        1. Discretize the trajectory, linearize at each point to get A, B, C matrices
        2. Solve the constrained optimization problem using these A, B, and C matrices to get a new trajectory, T_new
        3. T_curr = T_new
    """
    state_size = initial_state.shape[0]
    control_size = vehicle.action_space_size

    vehicle.propogateVehicleTrajectory(controls_guess)

    state_traj_curr = vehicle.state_trajectory[-50:]
    control_traj_curr = vehicle.control_trajectory[-50:]
    
    iter = 0
    last_value = np.inf
    curr_value = np.inf
    value_diff = -np.inf
    while value_diff < -tolerance and iter < max_iter:
        As = []
        Bs = []
        Cs = []
        
        start_linearization = time.time()
        
        for i, (state, control) in enumerate(zip(state_traj_curr, control_traj_curr)):
            # print(state.shape, state)
            # print(control.shape, control)
            if i % 5 == 0: #only linearize every 5 timesteps, and just use the same linearization for each
                A, B, C = vehicle.calculateLinearControlMatrices(state, control)
            As.append(A)
            Bs.append(B)
            Cs.append(C)

        end_linearization = time.time()

        start_optimization = time.time()

        optimization_prob, opt_states, opt_controls = initializeOptimizationProblem(time_horizon, As, Bs, Cs, initial_state, desired_state, state_size, control_size)

        optimization_prob.solve()

        if verbose:
            print(f"Linearization time: {end_linearization - start_linearization} seconds, optimization time: {time.time() - start_optimization} seconds")

        last_value = curr_value
        curr_value = optimization_prob.value
        value_diff = curr_value - last_value

        vehicle.resetVehicle()
        vehicle.propogateVehicleTrajectory(opt_controls.value)

        state_traj_curr = vehicle.state_trajectory
        control_traj_curr = vehicle.control_trajectory

        # print(f"Iteration {iter} complete. Value: {optimization_prob.value}")
        iter += 1

        # final_state_error = np.linalg.norm(state_traj_curr[-1,:] - desired_state)
    
    return opt_states.value, opt_controls.value
    
def optimize_trajectory(vehicle, initial_state, desired_state, dt=0.01, tolerance=0.01, verbose=False):
    final_state_error = np.inf

    time_horizon = 50
    
    state_size = initial_state.shape[0]
    control_size = vehicle.action_space_size

    #Initialize a random guess at the trajectory
    control_choices = np.array([0.0,5.0,10.0])
    controls_guess = np.random.choice(control_choices, size=(time_horizon,control_size))

    action_num = 1

    start = time.time()

    actions_to_take_btwn_opt = 3
    while final_state_error > tolerance:
        end_char = "\r" if not verbose else ""
        print(f"Finding optimal action {action_num}, with final error {final_state_error:.2f}:", end=end_char)
        vehicle_copy = copy.deepcopy(vehicle)

        max_iter = 1 if action_num > 1 else 10

        state_traj, control_traj = find_optimal_action(vehicle_copy, initial_state, desired_state, controls_guess, time_horizon, dt=dt, tolerance=0.01, max_iter=max_iter, verbose=verbose)

        final_state_error = np.linalg.norm(state_traj[-1,:] - desired_state)

        if verbose:
            print(f"Final state {state_traj[-1,:]}\n Final State error: {final_state_error}")
            print(f"Selected action: {control_traj[0,:]}")

        for action_to_take in range(actions_to_take_btwn_opt):
            vehicle.propagateVehicleState(control_traj[action_to_take,:])

        initial_state = vehicle.state

        controls_guess = jnp.vstack((control_traj[1:,:], np.zeros((actions_to_take_btwn_opt,control_size))))

        action_num += actions_to_take_btwn_opt

    print(f"Total optimization time: {time.time() - start} seconds")

    return vehicle
    # make_video("traj_opt_test", vehicle.state_trajectory, vehicle.control_trajectory, vehicle.thruster_positions, vehicle.thruster_force_vectors, dt, desired_state=desired_state)