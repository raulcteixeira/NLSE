import numpy as np
from scipy.constants import c, epsilon_0
from scipy.optimize import curve_fit
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
k_p = 2*np.pi*2.0e4 # perturbation k for measuring Bogoliubov dispersion
amp_p = 0.1 #amplitude of perturbation

def fit_func(x, a, b, c, freq, phase):
    return a * x + b*np.cos(2*np.pi*freq * x + phase) + c


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
    E_0 = (1 + amp_p*np.cos(2*k_p*simu.XX)).astype(
        PRECISION_COMPLEX
    )
    #simu.V = -1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / waist2**2).astype(
    #    PRECISION_COMPLEX
    #)

    # parameters for callback
    N_samples = 500
    z_samples = np.zeros(N_samples+1) # +1 to account for the 0th step
    z_samples[0] = 0
    E_samples = np.zeros((N_samples+1, N, N), dtype=np.complex64)
    E_0_normalized,_ = simu._prepare_output_array(E_0,True) 
    E_samples[0] = E_0_normalized.get()
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
            E_samples[i//save_every+1] = A.get()
            z_samples[i//save_every+1] = z
 
    A_plot = simu.out_field(E_0, L, callback=callback_sample, callback_args=(E_samples, z_samples), verbose=True)
    #A_plot = simu.out_field(E_0, L, verbose=True, plot=True, precision="single")

    # make fft of z evolution of the density at the center of the beam
    cuts_1D = abs(E_samples[:,int(simu.NX/2),:])** 2 * 1e-4 * c / 2 * epsilon_0
    center_cut = E_samples[:,int(simu.NX/2),int(simu.NY/2)]
    int_center_cut = abs(center_cut)** 2 * 1e-4 * c / 2 * epsilon_0
    center_cut_fft = np.abs(np.fft.fftshift(np.fft.fft(int_center_cut)))
    fft_freqs = np.fft.fftshift(np.fft.fftfreq(N_samples+1,save_every*simu.delta_z))

    # plot fft
    plt.figure()
    plot_int = 20
    plt.plot(fft_freqs[int(N_samples/2)-plot_int:int(N_samples/2)+plot_int],
        center_cut_fft[int(N_samples/2)-plot_int:int(N_samples/2)+plot_int])
    plt.savefig("img/temporal_fft.png")

    # obtain maximum of fft, excluding DC
    center_cut_fft_no_DC = center_cut_fft
    DC_cut_index = 3
    center_cut_fft_no_DC[int(N_samples/2)-DC_cut_index:int(N_samples/2)+DC_cut_index] = 0
    freq_index = np.argmax(center_cut_fft_no_DC)
    print([freq_index])
    print(fft_freqs[freq_index])

    # fit the z evolution to extract frequency
    plt.figure()
    plt.plot(z_samples,int_center_cut)

    popt, pcov = curve_fit(fit_func, z_samples, int_center_cut, p0 = [4,0.5,1.6,-1.*fft_freqs[freq_index],0])
    plt.plot(z_samples,fit_func(z_samples,*popt))
    plt.savefig("img/Z_evolution.png")

    print(popt[3])

    plt.figure()
    plt.imshow(cuts_1D)
    plt.savefig("img/1Dcut.png")

    plt.figure()
    #save the plot_field
    simu.plot_field(A_plot,L,True,"img/Output2.png")


if __name__ == "__main__":
    main()
