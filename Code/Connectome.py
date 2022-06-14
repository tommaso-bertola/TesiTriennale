import numpy as np

# example of variable attempts
#attempts = [10, 20, 100, 300, 500]
#attempts={0:1,10:10,15:10}
def connectome_attempter(attempts, mode,W):
    output=[[dict() for i in range(attempts[j])]for j in attempts]
    w_new=[np.zeros((attempts[j], 66,66)) for j in attempts]
    if mode=='add':
        for i,j in enumerate(attempts):
            w_new[i]=modify_connectome(W,attempts[j],j)
    if mode=='enhance':
        for i,j in enumerate(attempts):
            w_new[i]=enhance_connectome(W,attempts[j],j)
    return output,w_new


def modify_connectome(W, n_attempts, n_to_fill):
    # parameters
    n_neurons = W.shape[0]
    h_neurons = int(n_neurons/2)  # half the number of total neurons

    # array of all modified connectomes
    # this array will be returned by this function
    w_all = np.zeros((n_attempts, n_neurons, n_neurons), dtype=np.float64)

    # get only LR emisphere
    # it is a matrix n_neurons/2 x n_neurons/2
    lr_orig = lr_emisphere(W)

    # get the new distribution of weights
    w_n = weights_distribution(W)

    # tuples of empty indexes
    empty_coords = [(i, j)for i in range(h_neurons)
                    for j in range(h_neurons) if lr_orig[i, j] == 0]

    # create the modified connectomes
    for j in range(n_attempts):
        # check
        if n_to_fill > len(empty_coords):
            print('n_to_fill bigger than possible new weights')
            print(len(empty_coords)+' max limit of n_to_fill')
            return
        
        # shuffle to avoid repeating patterns on new connectome
        np.random.shuffle(empty_coords)
        np.random.shuffle(w_n)

        # lr to be modified
        lr_mod = np.copy(lr_orig)

        for i in range(n_to_fill):  # the number of empty places i want to fill
            lr_mod[empty_coords[i][0], empty_coords[i][1]] = w_n[i]

        # save the modified connectomes to w_all
        w_tmp = np.copy(W)
        w_tmp[0:h_neurons, h_neurons:] = lr_mod
        w_tmp[h_neurons:, 0:h_neurons] = lr_mod.T
        w_all[j] = w_tmp

    return w_all


def enhance_connectome(W, n_attempts):
    # parameters
    n_neurons = W.shape[0]
    h_neurons = int(n_neurons/2)  # half the number of total neurons
    # array of all modified connectomes
    # this array will be returned by this function
    w_all = np.zeros((n_attempts, n_neurons, n_neurons), dtype=np.float64)
    # get only LR emisphere
    # it is a matrix n_neurons/2 x n_neurons/2
    lr_orig = lr_emisphere(W)
    # tuples of not empty indexes
    not_empty_coords = [(i, j)for i in range(h_neurons)
                    for j in range(h_neurons) if lr_orig[i, j] != 0]
    # create the modified connectomes
    for j in range(n_attempts):

        # lr to be modified
        lr_mod = np.copy(lr_orig)
        for i in range(len(not_empty_coords)):  # the number of empty places i want to fill
            lr_mod[not_empty_coords[i][0], not_empty_coords[i][1]] = lr_orig[not_empty_coords[i][0], not_empty_coords[i][1]]*1.1
        # save the modified connectomes to w_all
        w_tmp = np.copy(W)
        w_tmp[0:h_neurons, h_neurons:] = lr_mod
        w_tmp[h_neurons:, 0:h_neurons] = lr_mod.T
        w_all[j] = w_tmp
    return w_all


# returns only the lr emisphere of the connectome
def lr_emisphere(W):
    n_neurons = W.shape[0]
    if n_neurons % 2 == 0:
        h_neurons=int(n_neurons/2)
        lr_orig = W[0:h_neurons, h_neurons:]
        return lr_orig
    else:
        print('Connectome with odd number of neurons')
        return

# generate random distribution following the original weights distribution
def weights_distribution(W, n_weights=1000):
    # prepare distribution
    # remove 0 entries
    w = W[W != 0].flatten()

    # compute histogram of distributions
    w_max = W.max()+0.01
    hist, binning = np.histogram(w, bins=50, range=(0, w_max), density=True)

    # compute cumulative distribution
    cdf = np.cumsum(hist/hist.sum())

    # array of random uniform numbers
    u = np.random.uniform(0, 1, n_weights)

    # new distribution of weights
    w_n = binning[np.searchsorted(cdf[:-1], u)]

    return w_n
