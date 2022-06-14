import numpy as np
from numba import jit, prange
from numba.typed import List
from scipy import signal
from Simulation import *
from Utilities import *
from Connectome import *


########################################################
# Object containing all important and parallel stuff   #
########################################################
class Brain:
    def __init__(self) -> None:
        pass

    def connectome(self, W, normalize=False):
        self.W = W
        self.n_neurons = W.shape[0]
        if normalize:
            self.normalize_connectome()
        else:
            print('Connectome loaded but not yet normalized')
        return self.W.shape[0]

    def normalize_connectome(self):
        self.W = self.W/self.W.sum(axis=1)[:, None]
        print('Connectome of shape '+str(self.W.shape) +
              ' now loaded and normalized successfully')

    def set_netowrk_parameters(self, r1, r2):
        self.r1 = r1
        self.r2 = r2
        print('r1 and r2 parameters now set successfully')

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


def advanced_simulation(w_new, brain, output, 
                        active_frac=0.1, n_runs=100,
                        tmin=0.001, tmax=0.3, delta_tc=0.1,
                        dt=0.1, n_timesteps=600,
                        compute_s1_s2=False, s_step=10,
                        compute_s_distrib=False, tc_distrib=0.15,
                        compute_fc=False,
                        ):
    for i in range(len(w_new)):
        for j in range(len(w_new[i])):
            brain.connectome(w_new[i][j], normalize=True)
            output[i][j] = brain.simulation(active_frac=active_frac, n_runs=n_runs,
                                            tmin=tmin, tmax=tmax, delta_tc=delta_tc,
                                            dt=dt, n_timesteps=n_timesteps,
                                            compute_s1_s2=compute_s1_s2, s_step=s_step,
                                            compute_s_distrib=compute_s_distrib, tc_distrib=tc_distrib,
                                            compute_fc=compute_fc)
    return output
