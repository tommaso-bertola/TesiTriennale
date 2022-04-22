import numpy as np
from numba import jit, prange


class Brain:
    def __init__(self, W) -> None:
        self.W = W
        self.n_neurons = W.shape[0]

    def set_netowrk_parameters(self, r1, r2, tc):
        self.r1 = r1
        self.r2 = r2
        self.tc = tc

    def simulation(self, active_frac=0.1, n_runs=100,
                   tmin=0.001, tmax=0.3, delta_tc=0.1,
                   dt=0.1, n_timesteps=600):

        n_neurons = self.n_neurons
        total_time = dt*n_timesteps

        # Array of the tested tc for each simulation
        tc = np.arange(tmin, tmax, delta_tc, dtype=np.float64)

        # Arrays containing activity, sigma activity, s1 and s2
        activity = np.zeros_like(tc, dtype=np.float64)
        sigma_activity = np.zeros_like(tc, dtype=np.float64)
        #s1 = s2 = np.zeros_like(tc, dtype=np.float64)

        # Init of random states for the simulation
        states_init = generate_initial_conf(active_frac=active_frac,
                                            n_neurons=n_neurons,
                                            n_runs=n_runs)
        # For every tc
        for i_tc, tc_testing in enumerate(tc):

            # Matrix containing activities
            activity_rtn = np.zeros(
                (n_runs, n_timesteps, n_neurons), dtype=np.float64)

            # Copy the random states
            states = states_init

            # Initial adjustment of the system
            for dummy in range(100):
                states, temp_act = update_states(states=states,
                                                 r1=self.r1, r2=self.r2, tc=tc_testing, W=self.W)

            # Real simulation
            # For each time step:
            for timestep in range(n_timesteps):

                # Save states and activity
                states, activity_rtn[:, timestep] = update_states(states=states,
                                                                  r1=self.r1, r2=self.r2, tc=tc_testing, W=self.W)

            At = np.mean(activity_rtn, axis=2)
            activity[i_tc] = np.mean(At)
            sigma_activity[i_tc] = np.mean(np.std(At, axis=1))

        # Return vector of tc and associated activities
        return tc, activity, sigma_activity  # , s1, s2

 ##################################


@jit(nopython=True)
def generate_initial_conf(active_frac, n_neurons, n_runs):
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


@jit(nopython=True)
def update_states(states, r1, r2, tc, W):
    # save the number of runs in total
    n_runs = states.shape[0]

    # temp states with same dims as states
    temp_states = np.zeros(states.shape, dtype=np.float64)

    # compute for each run the new activity status
    for run in prange(n_runs):
        temp_states[run] = update_neurons(states[run], r1, r2, tc, W)

    return temp_states, (temp_states == 1).astype(np.float64)




# Return a list of the new (not already checked) neurons connected to index neuron
def get_new_neighbours(red, index, l):
    neigh = np.arange(0, red.shape[0], 1)[red[index].astype(np.bool8)]
    mask = np.isin(neigh, l, invert=True)
    return neigh[mask]

# Update the list to keep track of the progress
def save(neigh, n_fam,  l, lll):
    l = l+neigh.tolist()
    lll[n_fam] += neigh.size
    return l, lll

# Recursive function to find the connected family starting from i neuron
def get_active_connected(red, i,  l, lll, n_fam):
    new_neigh = get_new_neighbours(red, i, l)
    if new_neigh.size != 0:
        l, lll = save(new_neigh, n_fam, l, lll)
        for j in new_neigh:
            l, lll = get_active_connected(red, j, l, lll, n_fam)
    return l, lll

# Returns a vector containing the dimensions of all the connected active regions
# Takes as input the reduced adjacent matrix of the active neurons
def get_families(reduced):
    n_active = len(reduced)
    n_fam = -1
    l = []  # list of already checked neurons
    lll = []  # dimensions of active-connected regions

    # Proper beginning of search, saving to new "family"
    for i in range(n_active):
        if i not in l:  # if node i is not already checked and saved in l, go on
            n_fam += 1
            l.append(i)
            lll.append(1)
            l, lll = get_active_connected(reduced, i, l, lll, n_fam)

    # Return the list
    return sorted(lll, reverse=True)
