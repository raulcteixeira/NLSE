import numpy as np
from scipy.constants import c, epsilon_0
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from NLSE import NLSE

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32


N = 2048
n2 = -6e-10
k_max = 2*np.pi*3e3 # maximum k vector of the beam profile
#waist = 2.23e-3
window = 12 * 2*np.pi/k_max
puiss = 1.0
Isat = 1e8  # saturation intensity in W/m^2
L = 400e-3
alpha = 0

# averaging over several realizations
N_real = 100

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

    # parameters for callback
    N_samples = 10
    z_samples = np.zeros(N_samples+1) # +1 to account for the 0th step
    z_samples[0] = 0
    E_samples = np.zeros((N_samples+1, N, N), dtype=np.complex64)

    #vector for keeping track of state norm
    norm_state = np.zeros(N_samples + 1)

    #array for average of final state fft
    fft_average = np.zeros([simu.NY,simu.NX])

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
    
    # Parameters for calculating initial state
    k_step = 2*np.pi/window
    zero_index = int(simu.NX/2)
    kX = np.linspace(-k_step*zero_index,k_step*(zero_index-1),simu.NX)
    kY = kX
    kXX,kYY = np.meshgrid(kX,kY)

    # vectors to help in radial average of n(k)
    y, x = np.indices(kXX.shape)
    kk = np.sqrt((x - zero_index)**2 + (y - zero_index)**2)
    kk = kk.astype(np.int32)  # bin by integer radius
    nk = np.bincount(kk.ravel())
    k_size = nk.shape[0]
    k_vec = np.linspace(k_step,(k_size-1)*k_step,k_size)

    #vector to keep averaged n(k)
    radial_mean_fft = np.zeros([N_samples+1,nk.shape[0]])

    for jj in np.arange(N_real):
        print(jj+1)
        #E_0_fft = np.heaviside(k_max**2-kXX**2-kYY**2,1)*np.exp(1j*np.random.uniform(0,2*np.pi,[simu.NX,simu.NX]))
        E_0_fft = np.exp(-(kXX**2+kYY**2)/k_max**2)*np.exp(1j*np.random.uniform(0,2*np.pi,[simu.NX,simu.NX]))

        E_0 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E_0_fft))).astype(
            PRECISION_COMPLEX
        )

        E_0_normalized,_ = simu._prepare_output_array(E_0,True) 
        E_samples[0] = E_0_normalized.get()
    
        A_plot = simu.out_field(E_0, L, callback=callback_sample, callback_args=(E_samples, z_samples), verbose=True)
    
        #A_plot = simu.out_field(E_0, L, verbose=True, plot=True, precision="single")

        #obtain frequency content and norm of fields
        E_fft = np.zeros([N_samples+1,A_plot.shape[0],A_plot.shape[1]])
        E_fft[:] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_samples[:]))).astype(
            PRECISION_COMPLEX
        )

        amp_E_sqr = abs(E_samples)**2
        norm_state = norm_state + np.sum(np.sum(amp_E_sqr,axis=2),axis=1)
    
        E_fft_abs = abs(E_fft)
        fft_average = fft_average + E_fft_abs[N_samples]**2

        # Sum values in each radial bin
        for ii in np.arange(N_samples+1):
            E_out_fft_sqr = E_fft_abs[ii]**2
            tbin = np.bincount(kk.ravel(), E_out_fft_sqr.ravel())
            # accumulate in the radial_mean_fft
            radial_mean_fft[ii] = tbin / nk + radial_mean_fft[ii]

    radial_mean_fft = radial_mean_fft/N_real
    norm_state = norm_state/norm_state[0]
    fft_average = fft_average/N_real

    sum_n_k = np.sum(radial_mean_fft,1)
    kin_energy = np.sum(radial_mean_fft*k_vec**2,1)/sum_n_k

    #normalize to the norm of beam
    #radial_mean_fft = np.transpose(np.transpose(radial_mean_fft)/norm_state)

    #k_r = np.arange(len(nr))*k_step
    k_max_plot = 200*k_max
    plot_index = int(k_max_plot/k_step)

    # save data

    np.save("img/radial_mean_fft",radial_mean_fft)
    np.save("img/k_vec",k_vec)

    # plots
    plt.figure()
    for ii in np.arange(N_samples+1):
        plt.loglog(k_vec[0:plot_index],radial_mean_fft[ii,0:plot_index],label=(str("%.2f" % z_samples[ii])+" m"))

    exponent = -3
    in_min = 8
    in_max = 40
    plt.loglog(k_vec[in_min:in_max],1e29*(k_vec[in_min:in_max])**exponent,linestyle='--',label = r'1/k$^3$')
    exponent = -2
    in_min = 20
    in_max = 100
    plt.loglog(k_vec[in_min:in_max],1e24*(k_vec[in_min:in_max])**exponent,linestyle='--',label = r'1/k$^2$')
    plt.legend(loc = 'right', fontsize = 'small')
    fft_max = max(radial_mean_fft[0])
    plt.ylim([2e-6*fft_max,2*fft_max])
    plt.title("n(k) versus k")
    plt.ylabel("n(k)")
    plt.xlabel("k/(2$\pi$) (m$^{-1}$)")
    plt.savefig("img/azimuthal_fft_out.png")

    # plotting rescaled radial_mean_fft
    plt.figure()
    beta = 0.5
    dimension = 2
    alph = dimension*beta
    for ii in np.arange(N_samples+1):
        resc_k_vec = k_vec*(z_samples[ii]/z_samples[1])**beta
        resc_radial_mean_fft = radial_mean_fft/(z_samples[ii]/z_samples[1])**alph
        plt.loglog(resc_k_vec[0:plot_index],resc_radial_mean_fft[ii,0:plot_index],label=(str("%.2f" % z_samples[ii])+" m"))
    plt.ylim([2e-6*fft_max,2*fft_max])
    plt.legend(loc = 'right', fontsize = 'small')
    plt.title("n(k) versus k, rescaled")
    plt.ylabel("$n(k) / (z/z_{ref})^{\\alpha \pi}$")
    plt.xlabel("k/(2$\pi$)$\\times(z/z_{ref})^\\beta$ (m$^{-1}$)")
    plt.savefig("img/azimuthal_fft_out_resc.png")

    plt.figure()
    plt.imshow(abs(E_0_fft[zero_index-100:zero_index+100,zero_index-100:zero_index+100])**2)
    plt.title("zoom of n(k) vectoriel of input beam")
    plt.savefig("img/fft_input_beam.png")

    plt.figure()
    plt.imshow(abs(E_0)**2)
    plt.title("Intensity of input beam (1 realization)")
    plt.savefig("img/input_beam.png")

    plt.figure()
    plt.imshow(np.angle(E_0))
    plt.title("Phase of input beam (1 realization)")
    plt.savefig("img/phase_input_beam.png")

    plt.figure()
    plt.imshow(abs(E_samples[N_samples])**2)
    plt.title("Intensity of output beam (1 realization)")
    plt.savefig("img/output_beam.png")

    plt.figure()
    plt.imshow(np.angle(E_samples[N_samples]))
    plt.title("Phase of output beam (1 realization)")
    plt.savefig("img/phase_output_beam.png")
    
    plt.figure()
    E_out_fft = E_fft[N_samples]
    plt.imshow(fft_average[zero_index-100:zero_index+100,zero_index-100:zero_index+100])
    plt.title("zoom of n(k) vectoriel of output beam (averaged)")
    plt.savefig("img/fft_out_beam.png")

    plt.figure()
    for ii in np.arange(N_samples+1):
        plt.semilogy(k_vec,radial_mean_fft[ii],label=(str("%.2f" % z_samples[ii])+" m"))
    plt.legend(loc = 'right', fontsize = 'small')
    plt.title("n(k) versus k")
    plt.ylabel("n(k)")
    plt.xlabel("k/(2$\pi$) (m$^{-1}$)")
    plt.savefig("img/azimuthal_fft_out_1.png")

    plt.figure()
    plt.plot(z_samples,norm_state,label = "State norm")
    plt.plot(z_samples,norm_state[0]*np.exp(-alpha*z_samples),label = r'$\text{e}^{-\alpha z}')
    plt.legend()
    plt.xlabel("z(m)")
    plt.title("State norm")
    plt.savefig("img/state_norm.png")

    plt.figure()
    plt.plot(z_samples,kin_energy,label = "")
    plt.legend()
    plt.title("Average kinetic energy (arb. units)")
    plt.xlabel("z(m)")
    plt.savefig("img/kinetic_energy.png")

if __name__ == "__main__":
    main()