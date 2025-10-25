import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

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


def canvas_to_signal(canvas_data, num_points=1000):
    """
    Convert canvas drawing data to x,y arrays suitable for Fourier analysis.

    Parameters:
    -----------
    canvas_data : dict
        Canvas result from streamlit_drawable_canvas containing drawing objects
    num_points : int
        Number of points to sample the signal at (default: 1000)

    Returns:
    --------
    x : numpy array
        Time/x-axis values normalized to [0, 1]
    y : numpy array
        Signal amplitude values normalized to [-1, 1]
    """
    if canvas_data is None or canvas_data.json_data is None:
        return None, None

    objects = canvas_data.json_data.get("objects", [])
    if not objects:
        return None, None

    # Extract all path points from the canvas
    all_points = []
    for obj in objects:
        if obj["type"] == "path":
            path = obj["path"]
            for segment in path:
                if len(segment) >= 2:
                    # segment format: ["Q", x1, y1, x2, y2] or ["L", x, y] etc.
                    if segment[0] in ["L", "M"]:  # Line or Move
                        all_points.append([segment[1], segment[2]])
                    elif segment[0] == "Q":  # Quadratic curve
                        all_points.append([segment[3], segment[4]])

    if not all_points:
        return None, None

    # Convert to numpy array
    points = np.array(all_points)

    # Sort by x-coordinate
    sorted_indices = np.argsort(points[:, 0])
    points = points[sorted_indices]

    # Normalize x to [0, 1] range
    x_raw = points[:, 0]
    x_min, x_max = x_raw.min(), x_raw.max()
    if x_max - x_min < 1e-6:  # Avoid division by zero
        return None, None

    x_normalized = (x_raw - x_min) / (x_max - x_min)

    # Normalize y to [-1, 1] range (flip because canvas y increases downward)
    y_raw = points[:, 1]
    y_min, y_max = y_raw.min(), y_raw.max()
    if y_max - y_min < 1e-6:
        return None, None

    y_normalized = -((y_raw - y_min) / (y_max - y_min) * 2 - 1)

    # Interpolate to get uniform sampling
    x_uniform = np.linspace(0, 1, num_points)
    y_uniform = np.interp(x_uniform, x_normalized, y_normalized)

    return x_uniform, y_uniform


def calculate_fourier_coefficients(y, n_harmonics, period=1.0):
    """
    Calculate Fourier series coefficients (a0, an, bn) for an arbitrary signal.

    Parameters:
    -----------
    y : numpy array
        Signal values (assumed to be sampled uniformly over one period)
    n_harmonics : int
        Number of harmonics to calculate
    period : float
        Period of the signal (default: 1.0)

    Returns:
    --------
    a0 : float
        DC component (average value)
    an : numpy array
        Cosine coefficients (length: n_harmonics)
    bn : numpy array
        Sine coefficients (length: n_harmonics)
    """
    N = len(y)
    t = np.linspace(0, period, N, endpoint=False)

    # DC component (average value)
    a0 = np.mean(y)

    # Calculate an (cosine) and bn (sine) coefficients
    an = np.zeros(n_harmonics)
    bn = np.zeros(n_harmonics)

    omega = 2 * np.pi / period

    for n in range(1, n_harmonics + 1):
        # Using trapezoidal integration
        an[n-1] = 2 * np.trapz(y * np.cos(n * omega * t), t) / period
        bn[n-1] = 2 * np.trapz(y * np.sin(n * omega * t), t) / period

    return a0, an, bn


def fourier_series_arbitrary(t, a0, an, bn, period=1.0):
    """
    Generate Fourier series approximation from coefficients.

    Parameters:
    -----------
    t : numpy array
        Time values
    a0 : float
        DC component
    an : numpy array
        Cosine coefficients
    bn : numpy array
        Sine coefficients
    period : float
        Period of the signal (default: 1.0)

    Returns:
    --------
    numpy array
        Fourier series approximation
    """
    n_harmonics = len(an)
    result = a0 * np.ones_like(t)
    omega = 2 * np.pi / period

    for n in range(1, n_harmonics + 1):
        result += an[n-1] * np.cos(n * omega * t)
        result += bn[n-1] * np.sin(n * omega * t)

    return result

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