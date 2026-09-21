import numpy as np
import matplotlib.pyplot as plt

def svdplot(svalues, nmax=50, logscale=True):

    s_energy = (svalues**2) / np.sum(svalues**2)
    cum_energy = np.cumsum(s_energy)

    fig = plt.figure(figsize=(7,5))
    fig, ax = plt.subplots()

    if logscale:
        ax.semilogy(1 - cum_energy[:nmax], color = 'red')
    else:
        ax.plot(1 - cum_energy[:nmax], color = 'red')
    ax.set_title("Residual Energy")
    ax.set_ylabel("CumEnergy")
    ax.grid()
    ax.set_xlabel("nBasis")