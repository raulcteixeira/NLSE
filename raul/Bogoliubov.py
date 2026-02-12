import numpy as np
import matplotlib.pyplot as plt
from NLSE import NLSE

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32


N = 2048
n2 = -1.6e-9
waist = 2.23e-3
waist2 = 70e-6
window = 4 * waist
puiss = 0.01
Isat = 10e4  # saturation intensity in W/m^2
L = 10e-3
alpha = 20
k_p = 2*np.pi/(5e-4) # perturbation k for measuring Bogoliubov dispersion
amp_p = 0.1 #amplitude of perturbation


def main():
    simu = NLSE(
        alpha,
        puiss,
        window,
        n2,
        None,
        L,
        NX=N,
        NY=N,
        Isat=Isat,
        backend="GPU",
    )
    simu.delta_z = 0.5e-4
    E_0 = (np.exp(-(simu.XX**2 + simu.YY**2) / waist**2)*(1 + amp_p*np.sin(2*k_p*simu.XX))).astype(
        PRECISION_COMPLEX
    )
    #simu.V = -1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / waist2**2).astype(
    #    PRECISION_COMPLEX
    #)
    A_plot = simu.out_field(E_0, L, verbose=True, plot=True, precision="single")

    plt.figure()
    plt.plot(abs(A_plot[int(simu.NX/2),:]))
    plt.savefig("img/1Dcut.png")

    plt.figure()
    #save the plot_field
    simu.plot_field(A_plot,L,True,"img/Output2.png")


if __name__ == "__main__":
    main()
