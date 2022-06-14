import numpy as np
from numba import jit, prange
from numba.typed import List

############################################
# Analysis of active and connected neurons #
############################################


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


#########################################################################
# Load fMRI signal and compute FC empirical matrix with the right order #
#########################################################################

# Load data from specific subject
# Choose to read both blocks or not
def fmri_signal(subject=1, all_blocks=False):
    if all_blocks == True:
        a = np.loadtxt("../Data/fMRI/subj"+str(subject)+"_block1.txt")
        b = np.loadtxt("../Data/fMRI/subj"+str(subject)+"_block2.txt")
        return np.concatenate((a, b))
    elif all_blocks == False:
        a = np.loadtxt("../Data/fMRI/subj"+str(subject)+"_block1.txt")
        return a

# return the fc empirical matrix with the right order


def fc_empirical():
    # Getting ready for ordering of data using Hagmann way
    labels_ponce = np.loadtxt('../Data/fMRI/ROIs_Labels.txt', dtype=str)
    labels_hagmann = np.loadtxt("../Data/connectivity_matrix/centres.txt",
                                dtype=str, usecols=0)
    dic_ponce = {v: i for i, v in enumerate(labels_ponce)}
    # use this for the right order used in the simulations (Hagmann)
    # fmri signal uses the Ponce ordering
    new_order_ponce = [dic_ponce[i] for i in labels_hagmann]

    signals = np.zeros((24, 600, 66))
    for i in range(24):
        signals[i] = fmri_signal(i+1, all_blocks=True)

    new_signals = np.zeros_like(signals)
    for i in range(66):
        new_signals[:, :, i] = signals[:, :, new_order_ponce[i]]

    # Compute FC for all subjects (Hagmann order)
    correlations = np.zeros((24, 66, 66), dtype=np.float64)
    for subj in range(24):
        correlations[subj] = np.corrcoef(new_signals[subj], rowvar=False)

    # Take the mean of all 24 FC matrix
    corr = np.mean(correlations, axis=0)
    return corr


def set_to_zero(fc):
    n_neurons = fc.shape[0]
    # indexes of weights to be set to 0
    range_zeroes = [(n_neurons+1)*i for i in range(n_neurons)]
    fc_temp = np.copy(fc)
    for i in range_zeroes:
        fc_temp.flat[i] = 0

    # return the fc matrix with 0 in the diagonal
    return fc_temp


def remove_zeroes(fc):
    n_neurons = fc.shape[0]
    # indexes of weights to be set to 0
    range_zeroes = [(n_neurons+1)*i for i in range(n_neurons)]
    fc_temp = np.zeros_like(fc)
    fc_temp = np.delete(fc.flatten(), range_zeroes)
    # return the fc matrix with 0 in the diagonal
    return fc_temp

# Compute rho and chi now using only upper diagonal elements of fc matrix to avoid repetition
# you can still use all the matrix, but the diagonal elements will be removed from the computation


def rho_chi_added_weights(output, mode='upper'):
    n_added_weights = len(output)
    n_tc = output[0]['tc'].shape[0]
    n_bins = 50
    fc_emp = fc_empirical()

    if mode == 'upper':
        triu_indices = np.triu_indices(fc_emp.shape[0], 1)
        fc_emp = fc_emp[triu_indices]
    if mode == 'zeroes':
        fc_emp = remove_zeroes(fc_emp)
    if mode == 'set':
        fc_emp = set_to_zero(fc_emp)

    rho = np.zeros((n_added_weights, n_tc))
    chi = np.zeros((n_added_weights, n_tc))

    h_fmri, _ = np.histogram(fc_emp, bins=n_bins)
    h_fmri = h_fmri/h_fmri.sum()

    for w in range(n_added_weights):
        for i in range(n_tc):
            # fc_sim=remove_zeroes(output[w]['fc'][i])
            fc_sim = output[w]['fc'][i]

            if mode == 'upper':
                fc_sim = fc_sim[triu_indices]
            if mode == 'zeroes':
                fc_sim = remove_zeroes(fc_sim)
            if mode == 'set':
                fc_sim = set_to_zero(fc_sim)

            h_norm, _ = np.histogram(fc_sim, bins=n_bins)
            h_norm = h_norm/h_norm.sum()

            rho[w, i] = np.corrcoef(fc_emp, fc_sim)[1, 0]
            chi[w, i] = np.sqrt(np.nansum((h_fmri-h_norm)**2/(h_fmri+h_norm)))

    return rho, chi
