from fourier import *
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Fourier Series Visualizer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Fourier Series Visualizer")
st.markdown("Interactive visualization of Fourier series analysis")

# Add mode selector
mode = st.radio(
    "Choose mode:",
    ["Draw Your Signal", "Square Wave Demo"],
    horizontal=True
)

st.sidebar.header("⚙️ Parameters")

n_harmonics = st.sidebar.slider(
    "Number of Harmonics",
    min_value=1,
    max_value=100,
    value=10,
    help="More harmonics = better approximation"
)

if mode == "Draw Your Signal":
    # Canvas mode
    st.markdown("### Draw your signal below")
    st.markdown("Use your mouse or touchscreen to draw a signal. The app will calculate its Fourier series!")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Create canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=300,
            width=600,
            drawing_mode="freedraw",
            key="canvas",
        )

    with col2:
        st.markdown("#### Instructions:")
        st.markdown("""
        1. Draw a signal from left to right
        2. Try different shapes: sine waves, square waves, triangles, or anything!
        3. The Fourier series will approximate your drawing
        4. Adjust harmonics to see how it affects the approximation
        """)

    # Process canvas data
    if canvas_result.json_data is not None:
        x, y = canvas_to_signal(canvas_result, num_points=1000)

        if x is not None and y is not None:
            # Calculate Fourier coefficients
            a0, an, bn = calculate_fourier_coefficients(y, n_harmonics, period=1.0)

            # Generate approximation
            approximation = fourier_series_arbitrary(x, a0, an, bn, period=1.0)

            # Calculate error
            error = np.mean((y - approximation)**2)

            # Display results
            st.markdown("---")
            st.subheader("📈 Fourier Series Approximation")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Harmonics Used", n_harmonics)
            with col2:
                st.metric("Approximation Quality", f"{max(0, (1-error))*100:.1f}%")
            with col3:
                st.metric("DC Component (a₀)", f"{a0:.3f}")

            # Plot signal and approximation
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(x, y, label="Your Signal", linewidth=2, alpha=0.7, color='black')
            ax.plot(x, approximation, label=f'Fourier Approximation ({n_harmonics} harmonics)',
                    linewidth=2, color='#FF4B4B')
            ax.set_xlabel("Normalized Time", fontsize=12)
            ax.set_ylabel('Amplitude', fontsize=12)
            ax.set_title(f'Signal Reconstruction using {n_harmonics} harmonics',
                         fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-1.5, 1.5])
            st.pyplot(fig)

            # Plot frequency spectrum
            st.subheader("🎵 Frequency Spectrum")

            # Calculate magnitudes
            magnitudes = np.sqrt(an**2 + bn**2)
            harmonic_numbers = np.arange(1, n_harmonics + 1)

            fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 4))

            # Magnitude spectrum
            ax2.stem(harmonic_numbers, magnitudes, basefmt='')
            ax2.set_xlabel('Harmonic Number', fontsize=12)
            ax2.set_ylabel('Magnitude', fontsize=12)
            ax2.set_title('Magnitude Spectrum', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

            # Cosine and Sine coefficients
            x_pos = harmonic_numbers
            width = 0.35
            ax3.bar(x_pos - width/2, an, width, label='Cosine (aₙ)', alpha=0.8)
            ax3.bar(x_pos + width/2, bn, width, label='Sine (bₙ)', alpha=0.8)
            ax3.set_xlabel('Harmonic Number', fontsize=12)
            ax3.set_ylabel('Coefficient Value', fontsize=12)
            ax3.set_title('Fourier Coefficients', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            st.pyplot(fig2)
        else:
            st.info("👆 Draw a signal on the canvas above to see its Fourier series!")
    else:
        st.info("👆 Draw a signal on the canvas above to see its Fourier series!")

else:
    # Square Wave Demo mode
    frequency = st.sidebar.slider(
        "Signal Frequency (Hz)",
        min_value=1,
        max_value=10,
        value=2,
        help="How many cycles per second"
    )

    duration = 2.0
    sample_rate = 1000

    time = generate_time_array(duration, sample_rate)
    original = square_wave(time, frequency)
    approximation = fourier_series_square_wave(time, frequency, n_harmonics)

    error = np.mean((original-approximation)**2)

    st.metric(
        label="Approximation Quality",
        value=f"{(1-error)*100:.1f}%",
        help="Higher = better match to square wave"
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time, original, label="Target Square Wave", linewidth=2, alpha=0.7, color='black')
    ax.plot(time, approximation, label=f'Fourier Approximation ({n_harmonics} harmonics)',
            linewidth=2, color='#FF4B4B')
    ax.set_xlabel("Time (s)", fontsize=12)
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

    amplitudes = get_harmonic_apmplitudes(n_harmonics)
    harmonic_numbers = np.arange(1, n_harmonics+1)
    odd_amplitudes = amplitudes[harmonic_numbers%2==1]
    odd_harmonic_numbers = harmonic_numbers[harmonic_numbers%2==1]
    frequencies_hz = odd_harmonic_numbers * frequency

    fig2, ax2 = plt.subplots(figsize=(12, 4))
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

