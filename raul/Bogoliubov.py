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
puiss = 1.05
Isat = 10e4  # saturation intensity in W/m^2
L = 10e-3
alpha = 20


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
    E_0 = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(
        PRECISION_COMPLEX
    )
    #simu.V = -1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / waist2**2).astype(
    #    PRECISION_COMPLEX
    #)
    A_plot = simu.out_field(E_0, L, verbose=True, plot=True, precision="single")
    simu.plot_field(A_plot,L,True,"img/Output2.png")
    #Plot the pot_field

    # if A_plot.ndim > 2:
    #     while len(A_plot.shape) > 2:
    #         A_plot = A_plot[0]
    # #if (
    # #    self.__CUPY_AVAILABLE__
    # #    and isinstance(A_plot, cp.ndarray)
    # #    or self.__PYOPENCL_AVAILABLE__
    # #    and isinstance(A_plot, cla.Array)
    # #    ):
    # #        A_plot = A_plot.get()
    # fig, ax = plt.subplots(1, 3, layout="constrained", figsize=(15, 5))
    # fig.suptitle(rf"Field at $z$ = {L:.2e} m")
    # ext_real = [
    #     np.min(self.X) * 1e3,
    #     np.max(self.X) * 1e3,
    #     np.min(self.Y) * 1e3,
    #     np.max(self.Y) * 1e3,
    # ]
    # ext_fourier = [
    #     np.min(self.Kx) * 1e-3,
    #     np.max(self.Kx) * 1e-3,
    #     np.min(self.Ky) * 1e-3,
    #     np.max(self.Ky) * 1e-3,
    # ]
    # rho = np.abs(A_plot) ** 2 * 1e-4 * c / 2 * epsilon_0
    # phi = np.angle(A_plot)
    # im_fft = np.abs(np.fft.fftshift(np.fft.fft2(A_plot)))
    # im0 = ax[0].imshow(rho, extent=ext_real)
    # ax[0].set_title("Intensity")
    # ax[0].set_xlabel("x (mm)")
    # ax[0].set_ylabel("y (mm)")
    # fig.colorbar(im0, ax=ax[0], shrink=0.6, label=r"Intensity ($W/cm^2$)")
    # im1 = ax[1].imshow(
    #     phi,
    #     extent=ext_real,
    #     cmap="twilight_shifted",
    #     vmin=-np.pi,
    #     vmax=np.pi,
    # )
    # ax[1].set_title("Phase")
    # ax[1].set_xlabel("x (mm)")
    # ax[1].set_ylabel("y (mm)")
    # fig.colorbar(im1, ax=ax[1], shrink=0.6, label="Phase (rad)")
    # im2 = ax[2].imshow(
    #     im_fft,
    #     extent=ext_fourier,
    #     cmap="nipy_spectral",
    # )
    # ax[2].set_title("Fourier space")
    # ax[2].set_xlabel(r"$k_x$ ($mm^{-1}$)")
    # ax[2].set_ylabel(r"$k_y$ ($mm^{-1}$)")
    # fig.colorbar(im2, ax=ax[2], shrink=0.6, label="Intensity (a.u.)")
    # plt.savefig(output.png)


if __name__ == "__main__":
    main()
