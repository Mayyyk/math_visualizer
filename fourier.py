import scipy
import numpy as np
import matplotlib.pyplot as plt

def generate_time_array(duration, sample_rate): #s, Hz
    x = np.arange(0, duration+(1/sample_rate), 1/sample_rate)
    return np.array(x)

def square_wave(t, frequency):
    # y = []
    T = 1/frequency
    # for current_t in range(t):
    #     if (current_t%T) < T/2:
    #         y.append(1)
    #     else:
    #         y.append(0)
            
    y = np.where((t%T) < T/2, 1, -1)
    return np.array(y)

def fourier_series_square_wave(t, frequency, n_harmonics):
    amplitudes = get_harmonic_apmplitudes(n_harmonics)
    omega = 2*np.pi*frequency
    result = np.zeros_like(t)
    for n in range(1, n_harmonics+1):
        result+=amplitudes[n-1]*np.sin(n*omega*t)

    return result

def get_harmonic_apmplitudes(n_harmonics):
    amplitudes = []
    for n in range(1, n_harmonics+1):
        if n%2==1:
            amp = 4/(np.pi*n)
        else:
            amp = 0
        amplitudes.append(amp)
    return np.array(amplitudes)

time = generate_time_array(1, 1000)
y = square_wave(time, 5)

FREQUENCY = 2

original = square_wave(time, FREQUENCY)
approx_5 = fourier_series_square_wave(time, FREQUENCY, 5)
approx_50 = fourier_series_square_wave(time, FREQUENCY, 50)

plt.figure(figsize=(12, 4))
plt.plot(time, original, label='Original', linewidth=2)
plt.plot(time, approx_5, label='5 harmonics', alpha=0.7)
plt.plot(time, approx_50, label='50 harmonics', alpha=0.7)
plt.legend()
plt.show()