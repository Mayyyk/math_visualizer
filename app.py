from fourier import *
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import numpy as np

def extract_signal_from_drawing(image_data, num_points=1000):
    """
    Converts an image of a drawn signal into a 1D numpy array.
    
    Args:
        image_data: A numpy array representing the image (height, width, 4 - RGBA).
        num_points: The number of points to resample the signal to.
    
    Returns:
        A 1D numpy array representing the signal.
    """
    if image_data is None:
        return None

    # For uploaded images, we can't rely on the alpha channel.
    # Instead, we convert the image to grayscale and use the color intensity.
    # We use the red channel as a proxy for intensity, assuming a B&W drawing.
    gray_image = image_data[:, :, 0]
    # Find all non-black pixels (with a tolerance).
    drawn_pixels = np.where(gray_image > 20)

    if drawn_pixels[0].size == 0:
        return np.zeros(num_points)

    height, width, _ = image_data.shape

    # Create a raw signal array with the same width as the image.
    raw_signal = np.full(width, np.nan)
    
    # For each column in the image, find the vertical center of the drawn line.
    for x in range(width):
        y_coords = drawn_pixels[0][drawn_pixels[1] == x]
        if y_coords.size > 0:
            # We use the median of the y-coordinates to be robust to thickness and outliers.
            # The y-coordinate is inverted because image origin is top-left.
            raw_signal[x] = height - np.median(y_coords)

    # Fill in any gaps in the signal using linear interpolation.
    if np.isnan(raw_signal).any():
        x_coords = np.arange(width)
        not_nan = ~np.isnan(raw_signal)
        raw_signal = np.interp(x_coords, x_coords[not_nan], raw_signal[not_nan])

    # Resample the signal to the desired number of points for consistent analysis.
    resampled_signal = np.interp(
        np.linspace(0, width - 1, num_points),
        np.arange(width),
        raw_signal
    )

    # Normalize the signal to be in the range [-1, 1].
    min_val = np.min(resampled_signal)
    max_val = np.max(resampled_signal)
    if max_val > min_val:
        normalized_signal = 2 * (resampled_signal - min_val) / (max_val - min_val) - 1
    else:
        normalized_signal = np.zeros(num_points)
        
    return normalized_signal

st.set_page_config(
    page_title="Fourier Series Visualizer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Fourier Series Visualizer")

# Create tabs for different modes
tab1, tab2 = st.tabs(["Classic Square Wave", "🎨 Draw Your Own Signal"])

with tab1:
    st.markdown("Interactive visualization of how square waves are built from sine waves")

    st.sidebar.header("⚙️ Parameters")

    st.sidebar.markdown("""
    ### 🔍 Gibbs Phenomenon
    Notice the ~9% overshoot at jumps?  
    This **never disappears**, even with infinite harmonics!

    It's a fundamental property of Fourier Series.
    """)

    frequency = st.sidebar.slider(
        "Signal Frequency (Hz)",
        min_value=1,
        max_value=10,
        value=2,
        help="How many cycles per second"
    )

    n_harmonics = st.sidebar.slider(
        "Number of Harmonics",
        min_value=1,
        max_value=100,
        value=5,
        help="More harmonics = better approximation"
    )

    duration = 2.0  # pokazujemy 2 sekundy
    sample_rate = 1000

    time = generate_time_array(duration, sample_rate)
    original = square_wave(time, frequency)
    approximation = fourier_series_square_wave(time, frequency, n_harmonics)

    error = np.mean((original-approximation)**2) # mean squared error


    st.metric(
        label="Approximation Quality",
        value=f"{(1-error)*100:.1f}%",
        help="Higher = better match to square wave"
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(time, original, label="Target Square Wave.", linewidth=2, alpha = 0.7, color='black')
    ax.plot(time, approximation, label=f'Fourier Approximation ({n_harmonics} harmonics)', 
            linewidth=2, color='#FF4B4B')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(f'Square Wave Reconstruction using {n_harmonics} harmonics', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.5, 1.5])

    st.pyplot(fig)


    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 How it works")
    st.sidebar.markdown(f"""
    **Current setup:**
    - Using **{n_harmonics}** sine waves
    - Showing only odd harmonics
    - Frequency: **{frequency} Hz**

    **Try this:**
    - Increase harmonics → smoother square
    - Notice the "overshoot" (Gibbs phenomenon)
    """)

    st.subheader("🎵 Frequency Spectrum")

    amplitudes = get_harmonic_amplitudes(n_harmonics)
    harmonic_numbers = np.arange(1, n_harmonics+1)

    odd_amplitudes = amplitudes[harmonic_numbers%2==1]
    odd_harmonic_numbers = harmonic_numbers[harmonic_numbers%2==1]

    frequencies_hz = odd_harmonic_numbers * frequency

    fig2, ax2 = plt.subplots(figsize = (12, 4))
    ax2.stem(frequencies_hz, odd_amplitudes, basefmt='')
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Amplitude (bₙ)', fontsize=12)
    ax2.set_title('Fourier Coefficients', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.text(0.02, 0.95, f'Base frequency: {frequency} Hz', 
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    st.pyplot(fig2)

    if st.button("Play original signal sound."):
        play_signal(time, original, duration=2.0)
    if st.button("Play approximated signal sound."):
        play_signal(time, approximation, duration=2.0)

with tab2:
    st.markdown("### Upload an image of a signal and see its Fourier series approximation!")

    st.info("""
    Draw a signal in any image editor (like Paint, GIMP, etc.) on a black background with a white pen, and upload it here.
    The drawing should go from left to right.
    """)

    # Create a file uploader widget for image files.
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    # This block executes when a file has been uploaded.
    if uploaded_file is not None:
        # Open the uploaded image and convert it to RGBA format.
        image = Image.open(uploaded_file).convert('RGBA')
        # Convert the image to a numpy array for processing.
        image_data = np.array(image)

        # Add a slider to control the number of harmonics for the custom signal.
        n_harmonics_custom = st.slider(
            "Number of Harmonics",
            min_value=1,
            max_value=100,
            value=10,
            key="n_harmonics_custom",
            help="More harmonics = better approximation for your custom signal"
        )

        # Extract the 1D signal from the image data.
        num_points = 1000
        custom_signal = extract_signal_from_drawing(image_data, num_points=num_points)

        # Check if a valid signal was extracted.
        if custom_signal is not None and np.any(custom_signal):
            # Define the time array for the signal.
            duration = 2.0
            time = np.linspace(0, duration, num_points)

            # Calculate the Fourier series of the custom signal.
            approximation, an, bn = fourier_series_custom_signal(time, custom_signal, n_harmonics_custom)

            # Plot the original and approximated signals.
            st.subheader("📈 Signal Reconstruction")
            fig_custom, ax_custom = plt.subplots(figsize=(12, 5))
            ax_custom.plot(time, custom_signal, label="Your Drawn Signal", linewidth=2, alpha=0.7, color='black')
            ax_custom.plot(time, approximation, label=f'Fourier Approximation ({n_harmonics_custom} harmonics)',
                           linewidth=2, color='#FF4B4B')
            ax_custom.set_xlabel("Time (s)")
            ax_custom.set_ylabel('Amplitude')
            ax_custom.set_title(f'Custom Signal Reconstruction ({n_harmonics_custom} harmonics)',
                                fontweight='bold')
            ax_custom.legend()
            ax_custom.grid(True, alpha=0.3)
            ax_custom.set_ylim([-1.5, 1.5])
            st.pyplot(fig_custom)

            # Plot the frequency spectrum of the custom signal.
            st.subheader("🎵 Frequency Spectrum (Custom Signal)")
            # The amplitude of each harmonic is the square root of the sum of the squares of an and bn.
            amplitudes = np.sqrt(np.array(an)**2 + np.array(bn)**2)
            harmonic_numbers = np.arange(1, n_harmonics_custom + 1)
            base_frequency = 1 / duration
            frequencies_hz = harmonic_numbers * base_frequency

            fig_spec_custom, ax_spec_custom = plt.subplots(figsize=(12, 4))
            ax_spec_custom.stem(frequencies_hz, amplitudes, basefmt='')
            ax_spec_custom.set_xlabel('Frequency (Hz)')
            ax_spec_custom.set_ylabel('Amplitude')
            ax_spec_custom.set_title('Fourier Coefficients (Custom Signal)', fontweight='bold')
            ax_spec_custom.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig_spec_custom)

            # Add buttons to play the audio of the signals.
            st.subheader("🔊 Audio Playback")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Play Your Drawn Signal", key="play_custom"):
                    play_signal(time, custom_signal, duration=duration)
            with col2:
                if st.button("Play Approximated Signal", key="play_approx_custom"):
                    play_signal(time, approximation, duration=duration)
