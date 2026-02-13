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
puiss = 1.05
Isat = 10e4  # saturation intensity in W/m^2
L = 50e-3
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
    simu.delta_z = 0.1e-4
    
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

    A_plot = simu.out_field(E_0, L, verbose=True, plot=True, precision="single")

    #obtain frequency content of field
    E_out_fft = np.fft.fftshift(np.fft.fft2(A_plot))

    plt.figure()
    plt.imshow(abs(E_out_fft))
    #simu.plot_field(A_plot,L,True,"img/Output2.png")
    plt.savefig("img/fft_out_beam.png")

    y, x = np.indices(E_out_fft.shape)
    r = np.sqrt((x - zero_index)**2 + (y - zero_index)**2)
    r = r.astype(np.int32)  # bin by integer radius

    E_out_fft_abs = abs(E_out_fft)
    
    # Sum values in each radial bin
    tbin = np.bincount(r.ravel(), E_out_fft_abs.ravel())
    nr = np.bincount(r.ravel())
    
    radial_mean = tbin / nr

    k_r = np.arange(len(radial_mean))*k_step
    plt.figure()
    plt.semilogy(k_r,radial_mean)
    plt.savefig("img/azimuthal_fft_out.png")

if __name__ == "__main__":
    main()