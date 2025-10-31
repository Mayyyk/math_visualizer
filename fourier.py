import scipy
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd


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
    amplitudes = get_harmonic_amplitudes(n_harmonics)
    omega = 2*np.pi*frequency
    result = np.zeros_like(t)
    for n in range(1, n_harmonics+1):
        result+=amplitudes[n-1]*np.sin(n*omega*t)

    return result
def fourier_series_custom_signal(t, custom_signal, n_harmonics):
    """
    Calculates the Fourier series for a custom signal.
    Returns the reconstructed signal, and the an, bn coefficients.
    """
    # The period of the signal is the total duration of the time array.
    dt = t[1] - t[0]
    T = t[-1] - t[0]
    if T == 0:
        return np.zeros_like(custom_signal), [], []

    # The fundamental angular frequency (omega) is 2*pi / T.
    omega = 2 * np.pi / T
    
    # Calculate Fourier coefficients using numerical integration.
    # a0 is the DC component (average value of the signal).
    a0 = (2 / T) * np.sum(custom_signal * dt)
    
    # an and bn are the coefficients for the cosine and sine terms, respectively.
    an = []
    bn = []
    for n in range(1, n_harmonics + 1):
        # For each harmonic, calculate the an and bn coefficients.
        cos_term = np.cos(n * omega * t)
        sin_term = np.sin(n * omega * t)
        an.append((2 / T) * np.sum(custom_signal * cos_term * dt))
        bn.append((2 / T) * np.sum(custom_signal * sin_term * dt))
    
    # Reconstruct the signal from the Fourier coefficients.
    # The reconstructed signal is the sum of the DC component and all harmonics.
    result = np.full_like(t, a0 / 2)
    for n in range(1, n_harmonics + 1):
        result += an[n-1] * np.cos(n * omega * t) + bn[n-1] * np.sin(n * omega * t)
        
    return result, an, bn

def get_harmonic_amplitudes(n_harmonics):
    amplitudes = []
    for n in range(1, n_harmonics+1):
        if n%2==1:
            amp = 4/(np.pi*n)
        else:
            amp = 0
        amplitudes.append(amp)
    return np.array(amplitudes)

def play_signal(time_array, signal_array, duration = 2.0):
    """
    Generates and plays an audio signal.
    """
    # Define a standard sample rate for audio playback.
    sample_rate = 44100
    
    # To play the signal, we need to resample it to the audio sample rate.
    # np.interp creates a new signal with the desired number of samples.
    audio_signal = np.interp(
        np.linspace(0, duration, int(sample_rate*duration)), # New time points for audio
        time_array,  # Original time points
        signal_array # Original signal values
        )
    
    # Normalize the audio signal to be between -1 and 1 to prevent clipping.
    audio_signal /= np.max(np.abs(audio_signal))
    
    # Play the audio signal.
    sd.play(audio_signal, sample_rate)
    sd.wait() # Wait for the sound to finish playing.
    
    

# time = generate_time_array(1, 1000)
# y = square_wave(time, 5)

# FREQUENCY = 2

# original = square_wave(time, FREQUENCY)
# approx_5 = fourier_series_square_wave(time, FREQUENCY, 5)
# approx_50 = fourier_series_square_wave(time, FREQUENCY, 50)

# plt.figure(figsize=(12, 4))
# plt.plot(time, original, label='Original', linewidth=2)
# plt.plot(time, approx_5, label='5 harmonics', alpha=0.7)
# plt.plot(time, approx_50, label='50 harmonics', alpha=0.7)
# plt.legend()
# plt.show()