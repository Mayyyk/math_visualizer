from fourier import *
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Fourier Series Visualizer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Fourier Series Visualizer")
st.markdown("Interactive visualization of how square waves are built from sine waves")

st.sidebar.header("⚙️ Parameters")

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
- Frequency: **{frequency} Hz**

**Try this:**
- Increase harmonics → smoother square
- Notice the "overshoot" (Gibbs phenomenon)
""")