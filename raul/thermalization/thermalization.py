import numpy as np
from scipy.constants import c, epsilon_0
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from NLSE import NLSE

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32


N = 2048
n2 = -1.6e-9
k_max = 2*np.pi*0.5e3 # maximum k vector of the beam profile
#waist = 2.23e-3
window = 8 * 2*np.pi/k_max
puiss = 1.0
Isat = 10e4  # saturation intensity in W/m^2
L = 500e-3
alpha = 2

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
    simu.delta_z = 0.2e-4
    
    # Calculation of E_0 with a square distribution of k's, with random phase
    # it works for even NX=NY; not sure for odd.
    k_step = 2*np.pi/window
    zero_index = int(simu.NX/2)
    kX = np.linspace(-k_step*zero_index,k_step*(zero_index-1),simu.NX)
    kY = kX
    kXX,kYY = np.meshgrid(kX,kY)
    E_0_fft = np.heaviside(k_max**2-kXX**2-kYY**2,1)*np.exp(1j*np.random.uniform(0,2*np.pi,[simu.NX,simu.NX]))

    plt.figure()
    plt.imshow(abs(E_0_fft[zero_index-100:zero_index+100,zero_index-100:zero_index+100]))
    plt.savefig("img/fft_input_beam.png")

    E_0 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E_0_fft))).astype(
        PRECISION_COMPLEX
    )

    plt.figure()
    plt.imshow(abs(E_0))
    plt.savefig("img/input_beam.png")

    # parameters for callback
    N_samples = 10
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
        if (i+1) % save_every == 0:
            E_samples[i//save_every+1] = A.get()
            z_samples[i//save_every+1] = z
 
    A_plot = simu.out_field(E_0, L, callback=callback_sample, callback_args=(E_samples, z_samples), verbose=True)
    
    #A_plot = simu.out_field(E_0, L, verbose=True, plot=True, precision="single")

    #obtain frequency content of fields
    E_fft = np.zeros([N_samples+1,A_plot.shape[0],A_plot.shape[1]])
    E_fft[:] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_samples[:]))).astype(
        PRECISION_COMPLEX
    )

    E_out_fft = E_fft[N_samples]
    plt.figure()
    plt.imshow(abs(E_out_fft))
    #simu.plot_field(A_plot,L,True,"img/Output2.png")
    plt.savefig("img/fft_out_beam.png")

    y, x = np.indices(E_out_fft.shape)
    r = np.sqrt((x - zero_index)**2 + (y - zero_index)**2)
    r = r.astype(np.int32)  # bin by integer radius
    nr = np.bincount(r.ravel())
    
    E_fft_abs = abs(E_fft)
    
    radial_mean_fft = np.zeros([N_samples+1,nr.shape[0]])

    # Sum values in each radial bin
    for ii in np.arange(N_samples+1):
        E_out_fft_abs = E_fft_abs[ii]
        tbin = np.bincount(r.ravel(), E_out_fft_abs.ravel())
        radial_mean_fft[ii] = tbin / nr

    #print(radial_mean_fft[10])
    
    k_r = np.arange(len(nr))*k_step
    k_max_plot = 200*k_max
    plot_index = int(k_max_plot/k_step)
    plt.figure()
    for ii in np.arange(N_samples+1):
        plt.loglog(k_r[0:plot_index],radial_mean_fft[ii,0:plot_index],label=(str("%.2f" % z_samples[ii])+" m"))

    #exponent = -8
    #in_min = 20
    #in_max = 100
    #plt.loglog(k_r[in_min:in_max],(k_r[in_min:in_max]/k_max/10)**exponent)
    plt.legend()
    plt.savefig("img/azimuthal_fft_out.png")

    plt.figure()
    plt.imshow(abs(E_samples[10]))
    plt.savefig("img/output_beam.png")


    plt.figure()
    for ii in np.arange(N_samples+1):
        plt.semilogy(k_r,radial_mean_fft[ii],label=(str("%.2f" % z_samples[ii])+" m"))
    plt.legend()
    plt.savefig("img/azimuthal_fft_out_1.png")

if __name__ == "__main__":
    main()