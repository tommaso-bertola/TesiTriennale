import numpy as np
from numba import jit, prange
from numba.typed import List
from scipy import signal


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
                   compute_s1_s2=False, s_step=10,
                   compute_s_distrib=False, tc_distrib=0.15,
                   compute_fc=False):

        n_neurons = self.n_neurons
        total_time = dt*n_timesteps

        # Array of the tested tc for each simulation
        tc = np.arange(tmin, tmax, delta_tc, dtype=np.float64)

        # Arrays containing activity, sigma activity, s1 and s2
        fc_matrix = np.zeros(
            (tc.shape[0], n_neurons, n_neurons), dtype=np.float64)
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
                s_dist = np.zeros(
                    (n_timesteps//s_step, n_runs * n_neurons), dtype=np.float64)

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

                # for the spcified timesteps
                if timestep % s_step == 0:
                    # save s1 and s2
                    if compute_s1_s2:
                        s1_t[timestep//s_step], \
                            s2_t[timestep // s_step],\
                            s_dist[timestep //
                                   s_step] = get_conn_comp(self.W, temp_active)
                        if compute_s_distrib:
                            if tc_testing < tc_distrib+0.001 and tc_distrib-0.001 <= tc_testing:
                                s_distrib = s_dist[s_dist > 0].flatten()

            At = np.mean(activity_rtn, axis=2)
            activity[i_tc] = np.mean(At)
            sigma_activity[i_tc] = np.mean(np.std(At, axis=1))
            if compute_s1_s2:
                s1[i_tc] = np.mean(s1_t)
                s2[i_tc] = np.mean(s2_t)
            if compute_fc:
                fc_matrix[i_tc] = self.get_fc_matrix(activity_rtn, dt, n_runs)
            del activity_rtn

        return_dic = dict()
        return_dic['tc'] = tc
        return_dic['activity'] = activity
        return_dic['sigma_activity'] = sigma_activity
        if compute_s_distrib:
            return_dic['s_distrib'] = s_distrib
        if compute_s1_s2:
            return_dic['s1'] = s1
            return_dic['s2'] = s2
        if compute_fc:
            return_dic['fc'] = fc_matrix

        # Return what was calculated
        return return_dic

    def get_fc_matrix(self, activity_rtn, dt, n_runs,
                      low_freq=0.01, high_freq=0.1,
                      ntaps=115):

        # Parameters
        n_neurons = self.n_neurons
        sample_rate = 1/dt
        nyquist_freq = sample_rate/2
        ntaps = ntaps
        band = [low_freq/nyquist_freq, high_freq/nyquist_freq]
        delay = 0.5 * (ntaps-1) / sample_rate

        # HRF coefficients
        a1 = 6
        a2 = 12
        b = 0.9
        c = 0.35
        d1 = a1*b
        d2 = a2*b

        # HRF function
        # the hrf is sampled only for the first 25 seconds at a sample frequency of 10 Hz
        t = np.arange(250)/sample_rate
        hrf = ((t/d1)**a1)*np.exp(-(t-d1)/b)-c*((t/d2)**a2)*np.exp(-(t-d2)/b)

        # FIR filter
        filt_fir = signal.firwin(ntaps, band,
                                 pass_zero=False, window='blackmanharris')

        # Convolve activity with hrf
        convolved_signals = np.array(
            [np.array(
                [np.convolve(activity_rtn[run, :, neuron], hrf, mode='same') for neuron in range(n_neurons)]) for run in range(n_runs)])

        # Apply fir filter to convolved signal
        filtered_signals = np.array(
            [np.array(
                [np.convolve(convolved_signals[run, neuron, :], filt_fir, mode='valid') for neuron in range(n_neurons)]) for run in range(n_runs)])

        # Compute FC matrix for each run.
        # returns any array of (n_neurons,n_neurons)
        fc_filtered_signals = np.array(
            [np.corrcoef(filtered_signals[i]) for i in range(n_runs)])

        # Take the mean of all n_runs as if they were coming from different people
        return np.mean(fc_filtered_signals, axis=0)

##################################


def get_sizes_distribution(s_distrib):
    x = np.arange(1, int(s_distrib.max())+1)
    l = np.zeros(int(s_distrib.max()))
    for i in range(int(s_distrib.max())):
        l[i] = ((s_distrib == i+1).astype(np.int16)).sum()
    ll = l/l.sum()
    return (x, ll)


@jit(nopython=True)
def get_cluster(reduced, checked, n, temp_cluster_elements, n_neurons):
    temp_cluster_elements.append(n)  # save the neuron n
    checked[n] = True  # confirm it was checked

    nearest = List()
    nearest.append(0)
    nearest.remove(0)
    for m in range(n_neurons):
        if (n != m) and reduced[n, m] > 0:  # find the nearest neighbours
            nearest.append(m)

    for m in nearest:
        if checked[m] == False:
            temp_cluster_elements = get_cluster(
                reduced, checked, m, temp_cluster_elements, n_neurons)

    return temp_cluster_elements


@jit(nopython=True)
def get_cc(W, n_neurons, active):
    reduced = (W*active).T*active

    checked = np.zeros(n_neurons, dtype=np.bool_)
    connected_comp = np.zeros(n_neurons)

    for n in range(n_neurons):
        if checked[n] == False:  # if not already checked
            if not active[n] > 0:
                continue
            temp_cluster_elements = List()
            temp_cluster_elements.append(0)
            temp_cluster_elements.remove(0)
            # find the list of the neurons connected to n
            cluster_elements = get_cluster(
                reduced, checked, n, temp_cluster_elements, n_neurons)
            connected_comp[n] = len(cluster_elements)

    # return sizes of clusters
    return -np.sort(-connected_comp)


@jit(nopython=True, parallel=True)
def get_conn_comp(W, active):
    n_runs, n_neurons = active.shape
    s1_r = np.zeros(n_runs, dtype=np.float64)
    s2_r = np.zeros(n_runs, dtype=np.float64)
    s_dist = np.zeros((n_runs, n_neurons), dtype=np.float64)

    for r in prange(n_runs):  # analyze every run
        s_dist[r] = get_cc(W, n_neurons, active[r])
        s1_r[r] = s_dist[r, 0]
        s2_r[r] = s_dist[r, 1]

    # return s1 and s2 for specific timestep
    return np.mean(s1_r), np.mean(s2_r), s_dist.flatten()


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