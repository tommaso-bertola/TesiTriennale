import numpy as np

class Brain:
    def __init__(self, W) -> None:
        self.W = W
        self.n_neurons = W.shape[0]

    def set_netowrk_parameters(self, r1, r2, tc):
        self.r1 = r1
        self.r2 = r2
        self.tc = tc

    def set_simulation_parameters(self, dt, n_timesteps, runs):
        self.dt = dt  # timestep
        self.n_timesteps = n_timesteps  # total number of steps in the simulation
        self.runs = runs  # parallel simulations fof random configs

    # def simulate(self, threshold_range=None):
    #    self.neurons_status = np.zeros((self.n_neurons, self.n_timesteps))
    #    # self.generate_initial_conf()

    def generate_initial_conf(self, active_fraction=0.3):
        """
        Generate initial *random* configurations of states
        for every neuron in the Brain and for every run
        """
        active_neurons = int(
            active_fraction*self.n_neurons)  # total numeber of active neurons
        # matrix with random configurations
        temp_states = np.zeros(self.n_neurons)

        temp_states[0:active_neurons] = 1  # set to 1 active neurons
        # set to -1 (refractary) the remaining half of neurons
        temp_states[-(self.n_neurons-active_neurons)//2:] = -1

        # create matrix to store initial confs
        self.states = np.zeros((self.runs,self.n_timesteps, self.n_neurons))
        # generate {runs} different states
        # shuffles the original temp_states distribution of active and refracatry neurons
        for run in range(self.runs):
            self.states[run,0] = np.random.choice(
                temp_states, self.n_neurons, replace=False)
            # (source, output array length, do not take twice)

    def compute_parameters(self):
        pass

    def update_neurons(self, state_neurons):

        # generate n_neurons random numbers
        p = np.random.random(self.n_neurons)

        # get the array of active neurons
        active_nodes = (state_neurons == 1).astype(np.float64)

        # compute the probability of becoming active for each neuron
        # and store it inside an array
        prob_active = self.r1+(1-self.r1)*(self.W@active_nodes > self.tc)

        return ((state_neurons == 1)*(-1) +                # active->inactive
                (state_neurons == 0)*(p < prob_active) +   # inactive->active following prob_active
                (state_neurons == -1)*(p > self.r2)*(-1))  # refractary->inactive following r2

    def update_status(self):
        print("Randomization")
        for run in range(self.runs):
            self.states[run,0] = self.update_neurons(self.states[run,0])
        print("End randomization")

        for run in range(self.runs):
            for time in range(1,self.n_timesteps):
                self.states[run,time]=self.update_neurons(self.states[run,time-1])
            print("End run "+str(run))
