import numpy as np
import matplotlib.pyplot as plt

def svdplot(svalues, nmax=50, logscale=True):

    s_energy = (svalues**2) / np.sum(svalues**2)
    cum_energy = np.cumsum(s_energy)

    fig = plt.figure(figsize=(7,5))
    fig, ax = plt.subplots()

    if logscale:
        ax.plot(np.log(1 - cum_energy[:nmax]), color = 'red')
        ax.set_title("Log-Cumulative Energy")
        ax.set_ylabel("LogCumEnergy")
    else:
        ax.plot(cum_energy[:nmax], color = 'red')
        ax.set_title("Cumulative Energy")
        ax.set_ylabel("CumEnergy")

    ax.set_xlabel("nBasis")