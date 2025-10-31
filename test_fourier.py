import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fourier import (
    generate_time_array,
    square_wave,
    fourier_series_square_wave,
    get_harmonic_amplitudes,
    play_signal
)


class TestGenerateTimeArray:
    """Tests for generate_time_array function"""

    def test_basic_time_array(self):
        """Test basic time array generation"""
        duration = 1.0
        sample_rate = 100
        result = generate_time_array(duration, sample_rate)

        assert len(result) == 101  # 0 to 1 second at 100 Hz = 101 points
        assert result[0] == 0.0
        assert np.isclose(result[-1], duration)

    def test_time_array_spacing(self):
        """Test that time points are evenly spaced"""
        duration = 2.0
        sample_rate = 1000
        result = generate_time_array(duration, sample_rate)

        # Check spacing between consecutive points
        differences = np.diff(result)
        expected_spacing = 1 / sample_rate
        assert np.allclose(differences, expected_spacing)

    def test_different_sample_rates(self):
        """Test with different sample rates"""
        duration = 1.0

        for sample_rate in [10, 100, 1000, 10000]:
            result = generate_time_array(duration, sample_rate)
            expected_length = int(duration * sample_rate) + 1
            assert len(result) == expected_length


class TestSquareWave:
    """Tests for square_wave function"""

    def test_square_wave_values(self):
        """Test that square wave alternates between 1 and -1"""
        t = np.linspace(0, 2, 1000)
        frequency = 1
        result = square_wave(t, frequency)

        # Check that values are only 1 or -1
        unique_values = np.unique(result)
        assert len(unique_values) == 2
        assert 1 in unique_values
        assert -1 in unique_values

    def test_square_wave_period(self):
        """Test that square wave has correct period"""
        frequency = 2
        period = 1 / frequency
        t = np.linspace(0, 1, 1000)
        result = square_wave(t, frequency)

        # First half of period should be 1, second half -1
        first_quarter = result[t < period/4]
        assert np.all(first_quarter == 1)

        third_quarter = result[(t > period/2) & (t < 3*period/4)]
        assert np.all(third_quarter == -1)

    def test_square_wave_multiple_frequencies(self):
        """Test square wave with different frequencies"""
        t = generate_time_array(1.0, 1000)

        for freq in [1, 2, 5, 10]:
            result = square_wave(t, freq)
            # Should still be binary values
            unique_values = np.unique(result)
            assert len(unique_values) == 2
            assert set(unique_values) == {1, -1}


class TestGetHarmonicAmplitudes:
    """Tests for get_harmonic_amplitudes function"""

    def test_odd_harmonics_nonzero(self):
        """Test that odd harmonics have non-zero amplitudes"""
        n_harmonics = 10
        amplitudes = get_harmonic_amplitudes(n_harmonics)

        # Odd harmonics (1, 3, 5, 7, 9) should be non-zero
        for n in range(1, n_harmonics + 1):
            if n % 2 == 1:
                assert amplitudes[n-1] > 0

    def test_even_harmonics_zero(self):
        """Test that even harmonics have zero amplitudes"""
        n_harmonics = 10
        amplitudes = get_harmonic_amplitudes(n_harmonics)

        # Even harmonics (2, 4, 6, 8, 10) should be zero
        for n in range(1, n_harmonics + 1):
            if n % 2 == 0:
                assert amplitudes[n-1] == 0

    def test_amplitude_formula(self):
        """Test that amplitudes follow the formula 4/(π*n) for odd n"""
        n_harmonics = 10
        amplitudes = get_harmonic_amplitudes(n_harmonics)

        for n in range(1, n_harmonics + 1):
            if n % 2 == 1:
                expected = 4 / (np.pi * n)
                assert np.isclose(amplitudes[n-1], expected)

    def test_amplitude_array_length(self):
        """Test that amplitude array has correct length"""
        for n in [1, 5, 10, 50, 100]:
            amplitudes = get_harmonic_amplitudes(n)
            assert len(amplitudes) == n

    def test_decreasing_amplitudes(self):
        """Test that odd harmonic amplitudes decrease"""
        n_harmonics = 20
        amplitudes = get_harmonic_amplitudes(n_harmonics)

        # Extract odd harmonics
        odd_amplitudes = [amplitudes[i] for i in range(0, n_harmonics, 2)]

        # Check they're decreasing
        for i in range(len(odd_amplitudes) - 1):
            assert odd_amplitudes[i] > odd_amplitudes[i + 1]


class TestFourierSeriesSquareWave:
    """Tests for fourier_series_square_wave function"""

    def test_basic_approximation(self):
        """Test basic Fourier series approximation"""
        t = generate_time_array(1.0, 1000)
        frequency = 2
        n_harmonics = 5

        result = fourier_series_square_wave(t, frequency, n_harmonics)

        assert len(result) == len(t)
        assert result.dtype == np.float64

    def test_single_harmonic(self):
        """Test with a single harmonic (fundamental frequency)"""
        t = generate_time_array(1.0, 1000)
        frequency = 1
        n_harmonics = 1

        result = fourier_series_square_wave(t, frequency, n_harmonics)

        # With 1 harmonic, should be a sine wave with amplitude 4/π
        expected_amplitude = 4 / np.pi
        assert np.max(result) <= expected_amplitude * 1.01  # Allow small tolerance
        assert np.min(result) >= -expected_amplitude * 1.01

    def test_more_harmonics_better_approximation(self):
        """Test that more harmonics give better square wave approximation"""
        t = generate_time_array(2.0, 1000)
        frequency = 2
        original = square_wave(t, frequency)

        approx_5 = fourier_series_square_wave(t, frequency, 5)
        approx_50 = fourier_series_square_wave(t, frequency, 50)

        # Calculate mean squared errors
        mse_5 = np.mean((original - approx_5) ** 2)
        mse_50 = np.mean((original - approx_50) ** 2)

        # More harmonics should have lower error
        assert mse_50 < mse_5

    def test_zero_harmonics(self):
        """Test edge case with zero harmonics"""
        t = generate_time_array(1.0, 100)
        frequency = 1
        n_harmonics = 0

        result = fourier_series_square_wave(t, frequency, n_harmonics)

        # Should be all zeros
        assert np.allclose(result, 0)

    def test_symmetry(self):
        """Test that Fourier approximation is symmetric around zero"""
        t = generate_time_array(2.0, 1000)
        frequency = 2
        n_harmonics = 20

        result = fourier_series_square_wave(t, frequency, n_harmonics)

        # Mean should be close to zero (symmetric)
        assert np.abs(np.mean(result)) < 0.1


class TestPlaySignal:
    """Tests for play_signal function"""

    @patch('fourier.sd.play')
    @patch('fourier.sd.wait')
    def test_play_signal_called(self, mock_wait, mock_play):
        """Test that sounddevice functions are called correctly"""
        t = generate_time_array(1.0, 1000)
        signal = square_wave(t, 2)
        duration = 1.0

        play_signal(t, signal, duration)

        # Check that play was called
        assert mock_play.called
        assert mock_wait.called

    @patch('fourier.sd.play')
    @patch('fourier.sd.wait')
    def test_signal_normalization(self, mock_wait, mock_play):
        """Test that signal is normalized before playing"""
        t = generate_time_array(1.0, 1000)
        signal = np.array([0, 5, -10, 3])  # Non-normalized signal

        play_signal(np.linspace(0, 1, 4), signal, duration=1.0)

        # Get the signal that was passed to play
        played_signal = mock_play.call_args[0][0]

        # Check that max absolute value is 1 (normalized)
        assert np.max(np.abs(played_signal)) <= 1.0

    @patch('fourier.sd.play')
    @patch('fourier.sd.wait')
    def test_correct_sample_rate(self, mock_wait, mock_play):
        """Test that correct sample rate is used"""
        t = generate_time_array(1.0, 1000)
        signal = square_wave(t, 2)

        play_signal(t, signal, duration=2.0)

        # Check sample rate argument
        call_args = mock_play.call_args
        assert call_args[0][1] == 44100  # Second argument should be sample rate


class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_full_pipeline(self):
        """Test the complete workflow from time array to approximation"""
        duration = 2.0
        sample_rate = 1000
        frequency = 2
        n_harmonics = 10

        # Generate time array
        t = generate_time_array(duration, sample_rate)

        # Generate original square wave
        original = square_wave(t, frequency)

        # Generate Fourier approximation
        approximation = fourier_series_square_wave(t, frequency, n_harmonics)

        # Basic sanity checks
        assert len(t) == len(original) == len(approximation)

        # Approximation should be within reasonable bounds
        assert np.max(approximation) < 2.0
        assert np.min(approximation) > -2.0

        # There should be some error but not too much
        mse = np.mean((original - approximation) ** 2)
        assert 0 < mse < 1.0

    def test_gibbs_phenomenon(self):
        """Test that Gibbs phenomenon (overshoot) is present"""
        t = generate_time_array(2.0, 2000)
        frequency = 1
        n_harmonics = 100  # Many harmonics

        approximation = fourier_series_square_wave(t, frequency, n_harmonics)

        # Gibbs phenomenon: overshoot should be about 9% above 1
        max_value = np.max(approximation)
        expected_overshoot = 1.09  # Approximately 9% overshoot

        # Allow some tolerance
        assert max_value > 1.05  # At least some overshoot
        assert max_value < 1.20  # But not too much
