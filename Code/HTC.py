import numpy as np
from numba import jit, prange
from numba.typed import List


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
                   dt=0.1, n_timesteps=600,
                   compute_s1_s2=False, s_step=10):

        n_neurons = self.n_neurons
        total_time = dt*n_timesteps

        # Array of the tested tc for each simulation
        tc = np.arange(tmin, tmax, delta_tc, dtype=np.float64)

        # Arrays containing activity, sigma activity, s1 and s2
        activity = np.zeros_like(tc, dtype=np.float64)
        sigma_activity = np.zeros_like(tc, dtype=np.float64)
        s1 = np.zeros_like(tc, dtype=np.float64)
        s2 = np.zeros_like(tc, dtype=np.float64)

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

            if compute_s1_s2:
                s1_t = np.zeros(n_timesteps//s_step,
                                dtype=np.float64)
                s2_t = np.zeros(n_timesteps//s_step,
                                dtype=np.float64)

            # Initial adjustment of the system
            for dummy in range(100):
                states, _ = update_states(states=states,
                                          r1=self.r1, r2=self.r2, tc=tc_testing, W=self.W)

            # Real simulation
            # For each time step:
            for timestep in range(n_timesteps):

                # Save states and activity
                states, temp_active = update_states(states=states,
                                                    r1=self.r1, r2=self.r2, tc=tc_testing, W=self.W)
                activity_rtn[:, timestep] = temp_active

                if compute_s1_s2:
                    if timestep % s_step == 0:
                        s1_t[timestep//s_step], s2_t[timestep //
                                                     s_step] = get_conn_comp(self.W, temp_active)

            At = np.mean(activity_rtn, axis=2)
            activity[i_tc] = np.mean(At)
            sigma_activity[i_tc] = np.mean(np.std(At, axis=1))
            if compute_s1_s2:
                s1[i_tc] = np.mean(s1_t)
                s2[i_tc] = np.mean(s2_t)
            else:
                s1 = s2 = 1

        del activity_rtn
        # Return vector of tc and associated activities
        return tc, activity, sigma_activity, s1, s2

 ##################################


@jit(nopython=True)
def get_cluster(reduced, checked, n, temp_cluster_elements, n_neurons):
    temp_cluster_elements.append(n)  # save the neuron n
    checked[n] = True  # confirm it was checked

    nearest = List()
    nearest.append(0)
    nearest.remove(0)
    for m in range(n_neurons):
        if (not n == m) and reduced[n, m] > 0:  # find the nearest neighbours
            nearest.append(m)

    for m in nearest:
        if not checked[m]:
            temp_cluster_elements = get_cluster(
                reduced, checked, m, temp_cluster_elements, n_neurons)

    return temp_cluster_elements


@jit(nopython=True)
def get_cc(W, n_neurons, active):
    reduced = (W*active).T*active

    checked = np.zeros(n_neurons, dtype=np.bool_)
    connected_comp = np.zeros(n_neurons)

    for n in range(n_neurons):
        if not checked[n]:  # if not already checked
            temp_cluster_elements = List()
            temp_cluster_elements.append(0)
            temp_cluster_elements.remove(0)
            # find the list of the neurons connected to n
            cluster_elements = get_cluster(
                reduced, checked, n, temp_cluster_elements, n_neurons)
            connected_comp[n] = len(cluster_elements)

    connected_comp = -np.sort(-connected_comp)  # sort the connected components

    # return s1 and s2 for the specific run
    return connected_comp[0], connected_comp[1]


@jit(nopython=True, parallel=True)
def get_conn_comp(W, active):
    n_runs, n_neurons = active.shape
    s1_r = np.zeros(n_runs, dtype=np.float64)
    s2_r = np.zeros(n_runs, dtype=np.float64)

    for r in prange(n_runs):  # analyze avaery run
        s1_r[r], s2_r[r] = get_cc(W, n_neurons, active[r])

    # return s1 and s2 for specific timestep
    return np.mean(s1_r), np.mean(s2_r)


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


if __name__ == "__main__":
    W = np.loadtxt("Data/connectivity_matrix/weights.txt")
    W = W/W.sum(axis=1)[:, None]
    brain = Brain(W)
    brain.set_netowrk_parameters(r1=2e-3, r2=0.288, tc=0.15)
    tc, a, sigma_a, s1, s2 = brain.simulation(active_frac=0.1, n_runs=100,
                                              tmin=0.005, tmax=0.2, delta_tc=0.005,
                                              dt=0.1, n_timesteps=600)
