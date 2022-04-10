import numpy as np
from numba import jit, prange


class Brain:
    def __init__(self, W) -> None:
        self.W = W
        self.n_neurons = W.shape[0]

    def set_netowrk_parameters(self, r1, r2, tc, tmin=0.01, tmax=3, delta_tc=0.1):
        self.r1 = r1
        self.r2 = r2
        self.tc = tc
        self.tmin = tmin
        self.tmax = tmax
        self.delta_tc = delta_tc

    def set_simulation_parameters(self, dt, n_timesteps, runs):
        self.dt = dt  # timestep
        self.n_timesteps = n_timesteps  # total number of steps in the simulation
        self.n_runs = runs  # parallel simulations fof random configs

    def simulation(self, active_frac=0.1, n_runs=100):
        n_neurons = self.W.shape[0]

        # Array of the tested tc for each simulation
        tc = np.arange(self.tmin, self.tmax, self.delta_tc, dtype=np.float64)

        # Array containing tc and associated activity
        activity = np.zeros_like(tc, dtype=np.float64)

        # Matrix containing activities of neurons for each run
        activity_t = np.zeros((self.n_timesteps), dtype=np.float64)

        # Init of random states for the simulation
        states_init = generate_initial_conf(active_frac, n_neurons, n_runs)

        # For every tc
        for i, t in enumerate(tc):

            # Copy the random states
            states = states_init

            # Init a temporary obj to store all the activity (for each run and neuron)
            temp_act = np.zeros(states.shape, dtype=np.float64)

            # Initial adjustment of the system
            for dummy in range(100):
                states, temp_act = update_states(
                    self.r1, self.r2, t, self.W, states)

            # Real simulation
            # For each time step:
            for time in range(self.n_timesteps):

                # Save states and activity
                states, temp_act = update_states(
                    self.r1, self.r2, t, self.W, states)

                # Compute activty at time t
                # Sum all 1s in the temp_act matrix
                # and divide by total number of neurons and runs
                activity_t[time] = ((temp_act.sum()).sum())/(n_neurons*n_runs)

            # Save the mean activity for the simulation at chosen tc
            activity[i] = activity_t.sum()/(self.dt*self.n_timesteps)

            print("End of ", t)

        # Return vector of tc and associated activities
        return (tc, activity)

 ##################################


@jit(nopython=True)
def generate_initial_conf(active_frac=0.5, n_neurons=998, n_runs=100):
    """
    Generate n_runs initial *random* configurations of states for every n_neuron 
    """
    # total numeber of active neurons
    active_neurons = int(active_frac*n_neurons)

    # init of array with random configurations
    temp_states = np.zeros(n_neurons, dtype=np.float64)

    # set to 1 active neurons
    # set to -1 (refractary) the remaining half of neurons
    temp_states[0:active_neurons] = 1
    temp_states[-(n_neurons-active_neurons)//2:] = -1

    # create matrix to store initial confs
    states = np.zeros((n_runs, n_neurons), dtype=np.float64)

    # generate n_runs different states
    # shuffles the original temp_states
    # distribution of active and refracatry neurons
    for run in prange(n_runs):
        states[run] = np.random.choice(temp_states, n_neurons, replace=False)
        # (source, output array length, do not take twice)

    return states


@jit(nopython=True)
def update_neurons(state_neurons, r1, r2, tc, W):
    # generate n_neurons random numbers
    p = np.random.random(state_neurons.shape[0])

    # get the array of active neurons
    active_nodes = (state_neurons == 1).astype(np.float64)

    # compute the probability of becoming active for each neuron
    # and store it inside an array
    prob_active = r1+(1-r1)*(W@active_nodes > tc)

    # active->inactive
    # inactive->active following prob_active
    # refractary->inactive following r2
    return ((state_neurons == 1)*(-1) +
            (state_neurons == 0)*(p < prob_active) +
            (state_neurons == -1)*(p > r2)*(-1))


@jit(nopython=True)  # , parallel=True)
def update_states(r1, r2, tc, W, states):
    # save the number of runs in total
    n_runs = states.shape[0]

    # temp states with same dims as states
    temp_states = np.zeros(states.shape, dtype=np.float64)

    # compute for each run the new activity status
    for run in prange(n_runs):
        temp_states[run] = update_neurons(states[run], r1, r2, tc, W)

    return temp_states, (temp_states == 1).astype(np.float64)
