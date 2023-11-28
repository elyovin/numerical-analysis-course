import numpy as np
import scipy.fft
import matplotlib.pyplot as plt

from typing_extensions import Self


class FourierTransform:
    def __init__(self) -> None:
        self.dft_solution: list = None
        self.fft_solution: list = None

    def get_spectral_density(self, x: np.ndarray) -> np.ndarray:
        return (x * np.conj(x)).real / len(x)

    def transform(self, x: np.ndarray, freq: int, n: int) -> Self:
        N = 2 ** n  # length of signal

        self.dft_solution = np.zeros(N, dtype=np.complex64)
        for i in range(N):
            pow_arr = -1j * 2 * np.pi / N * i * np.arange(N)
            self.dft_solution[i] = np.dot(x, np.exp(pow_arr))

        return self

    def fast_transform(self, x: np.ndarray, freq: int, n: int) -> Self:
        N = 2 ** n  # length of signal

        # Bit reverse order
        permutation = np.zeros(N).astype(int)
        for i in range(N):
            p = i
            for j in range(1, n + 1):
                permutation[i] += 2 ** (n - j) * (p - (p // 2) * 2)
                p //= 2
        self.fft_solution = x[permutation].astype(np.complex64)
        
        for k in range(1, n + 1):
            for i in range(2 ** (n - k)):
                for l in range(2 ** (k - 1)):
                    i1 = i * 2 ** k + l
                    i2 = i * 2 ** k + 2 ** (k - 1) + l
                    fft_1 = self.fft_solution[i1]
                    fft_2 = self.fft_solution[i2]
                    exp_val = np.exp(-2j * np.pi * l / (2 ** k)) 
                    x1 = fft_1 + exp_val * fft_2
                    x2 = fft_1 - exp_val * fft_2
                    self.fft_solution[i1] = x1
                    self.fft_solution[i2] = x2

        return self


def tabulate(n: int, freq: int) -> np.ndarray:
    t = np.arange(start=0, stop=2 ** n, step=1) / freq
    freq_1 = 50 * (10 % 3 + 1)
    freq_2 = 120 * (10 % 2 + 1)
    freq_3 = 30 * (10 % 5 + 1)
    x = (
        np.sin(2 * np.pi * freq_1 * t)
        + np.sin(2 * np.pi * freq_2 * t)
        + np.sin(2 * np.pi * freq_3 * t)
        + (10 % 3 + 1) * np.random.randn(len(t))
    )
    return x


def task() -> None:
    n = 9
    freq = 1000
    N = 2 ** n
    x = tabulate(n, freq)
    
    transformer = FourierTransform()
    transformer.transform(x, freq, n)
    transformer.fast_transform(x, freq, n)
    
    x_dft = transformer.dft_solution
    x_fft = transformer.fft_solution
    x_fft_check = scipy.fft.fft(x)
    x_plot = freq * np.arange(N // 2) / N

    fig, axs = plt.subplots(3, 3)

    # Plot original signal
    axs[0, 0].plot(x[:N])
    axs[0, 0].set_title('Исходный сигнал')
    
    # Plot real part of DFT
    axs[0, 1].plot(
        x_plot,
        x_fft_check.real[:N // 2]
    )
    axs[0, 1].set_title('Действительная часть ДПФ')

    # Plot imaginary part of DFT
    axs[0, 2].plot(
        x_plot,
        x_fft_check.imag[:N // 2] 
    )
    axs[0, 2].set_title('Мнимая часть ДПФ')

    # Plot implemented DFT
    axs[1, 0].plot(
        x_plot,
        transformer.get_spectral_density(x_dft)[:N // 2]
    )
    axs[1, 0].set_title('Реализованное ДПФ')

    # Plot implemented FFT
    axs[1, 1].plot(
        x_plot,
        transformer.get_spectral_density(x_fft)[:N // 2]
    )
    axs[1, 1].set_title('Реализованное БПФ')

    # Plot control FFT
    axs[1, 2].plot(
        x_plot,
        transformer.get_spectral_density(x_fft_check)[:N // 2]
    )
    axs[1, 2].set_title('Контрольное БПФ')
   
    # Plot inverse FFT of implemented DFT
    axs[2, 0].plot(
        scipy.fft.ifft(x_dft).real
    )
    axs[2, 0].set_title('Обратное БПФ \nреализованного ДПФ')

    # Plot inverse FFT of implemented DFT
    axs[2, 1].plot(
        scipy.fft.ifft(x_fft).real
    )
    axs[2, 1].set_title('Обратное БПФ \nреализованного БПФ')
    
    # Plot inverse FFT of control FFT
    axs[2, 2].plot(
        scipy.fft.ifft(x_fft_check).real
    )
    axs[2, 2].set_title('Обратное БПФ \nконтрольного БПФ')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    task()
