import numpy as np
from scipy.constants import c, epsilon_0
import matplotlib.pyplot as plt
from NLSE import NLSE

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32


N = 2048
n2 = -1.6e-19
waist = 2.23e-3
waist2 = 70e-6
window = 4 * waist
puiss = 1.05
Isat = 10e4  # saturation intensity in W/m^2
L = 50e-3
alpha = 2
k_p = 2*np.pi/(4e-4) # perturbation k for measuring Bogoliubov dispersion
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
    simu.delta_z = 0.1e-4
    #np.exp(-(simu.XX**2 + simu.YY**2) / waist**2)*
    E_0 = ((1 + amp_p*np.sin(2*k_p*simu.XX))).astype(
        PRECISION_COMPLEX
    )
    #simu.V = -1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / waist2**2).astype(
    #    PRECISION_COMPLEX
    #)

    # parameters for callback
    N_samples = 500
    z_samples = np.zeros(N_samples+1) # +1 to account for the 0th step
    E_samples = np.zeros((N_samples+1, N, N), dtype=np.complex64)
    N_steps = int(round(L/simu.delta_z))
    save_every = N_steps//N_samples



    def callback_sample(simu: NLSE, A: np.ndarray, z: float, i: int, E_samples: np.ndarray, z_samples: np.ndarray) -> None:
        """A callback function for the NLSE class.

        Args:
            simu (NLSE): The simulation object
            A (np.ndarray): The field at the current step
            z (float): The current propagation distance in meters
            i (int): The main loop index
            E (np.ndarray): The array of samples
        """
        if i % save_every == 0:
            E_samples[i//save_every] = A.get()
            z_samples[i//save_every] = z
 
    A_plot = simu.out_field(E_0, L, callback=callback_sample, callback_args=(E_samples, z_samples), verbose=True)
    #A_plot = simu.out_field(E_0, L, verbose=True, plot=True, precision="single")

    cuts_1D = abs(E_samples[:,int(simu.NX/2),:])** 2 * 1e-4 * c / 2 * epsilon_0

    plt.figure()
    plt.imshow(cuts_1D)
    plt.savefig("img/1Dcut.png")

    plt.figure()
    #save the plot_field
    simu.plot_field(A_plot,L,True,"img/Output2.png")


if __name__ == "__main__":
    main()
