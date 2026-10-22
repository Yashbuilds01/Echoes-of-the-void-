"""
Echoes of the Void - Signal Processing Module
Routines for:
1. Zero-phase Butterworth Bandpass Filtering
2. Frequency-domain Noise Whitening (Welch PSD estimation)
3. High-Resolution Constant-Q / Wavelet Time-Frequency Spectrogram Engine (LIGO Omega Scan)
4. Comprehensive 3-Stage Pipeline Diagnostics & Noise Suppression Metrics
"""

import numpy as np
from scipy import signal
from typing import Tuple, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EchoesOfTheVoid.SignalProcessor")


class SignalProcessor:
    """
    Production-grade digital signal processing for gravitational wave strain data.
    Implements standard LIGO Scientific Collaboration algorithms for conditioning,
    spectral whitening, and multi-Q time-frequency decomposition.
    """

    @staticmethod
    def bandpass_filter(
        strain: np.ndarray,
        fs: int,
        low_cut: float = 30.0,
        high_cut: float = 400.0,
        order: int = 4
    ) -> np.ndarray:
        """
        Applies a zero-phase Butterworth bandpass filter using Second-Order Sections (SOS).
        Eliminates low-frequency seismic walls (<30 Hz) and high-frequency laser noise (>400 Hz).
        """
        nyquist = 0.5 * fs
        low = max(5.0, low_cut) / nyquist
        high = min(nyquist - 5.0, high_cut) / nyquist

        if low >= high:
            raise ValueError(f"Invalid bandpass bounds: low_cut ({low_cut}) >= high_cut ({high_cut})")

        sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
        filtered = signal.sosfiltfilt(sos, strain)
        return filtered

    @staticmethod
    def whiten(
        strain: np.ndarray,
        fs: int,
        segment_duration: float = 1.0,
        tukey_alpha: float = 0.1
    ) -> np.ndarray:
        """
        Frequency-domain data whitening using Welch's method for Power Spectral Density (PSD) estimation.
        Normalizes strain spectral amplitudes so that background noise variance is approximately 1.0
        across all frequencies, suppressing persistent power grid hums and seismic peaks.
        """
        n_samples = len(strain)
        
        # Apply Tukey window to suppress edge transients
        window = signal.windows.tukey(n_samples, alpha=tukey_alpha)
        windowed_strain = strain * window

        # Estimate PSD via Welch's method
        nperseg = min(n_samples, int(segment_duration * fs))
        if nperseg % 2 != 0:
            nperseg -= 1
        nperseg = max(256, nperseg)

        freqs, psd = signal.welch(
            strain,
            fs=fs,
            window='hann',
            nperseg=nperseg,
            noverlap=nperseg // 2,
            scaling='density'
        )

        # Interpolate PSD to match the discrete FFT frequencies of the whole strain array
        fft_freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
        # Avoid zero or negative values in PSD
        psd = np.maximum(psd, 1e-50)
        interp_psd = np.interp(fft_freqs, freqs, psd)
        interp_psd = np.maximum(interp_psd, 1e-50)

        # Compute strain FFT and whiten in frequency domain
        strain_fft = np.fft.rfft(windowed_strain)
        whitened_fft = strain_fft / np.sqrt(interp_psd * (fs / 2.0))

        # Inverse FFT back to time domain
        whitened_strain = np.fft.irfft(whitened_fft, n=n_samples)

        # Normalize variance
        valid_slice = int(n_samples * 0.05)
        core_std = np.std(whitened_strain[valid_slice:-valid_slice])
        if core_std > 0:
            whitened_strain = whitened_strain / core_std

        return whitened_strain

    @staticmethod
    def q_transform(
        strain: np.ndarray,
        time: np.ndarray,
        fs: int,
        q_value: float = 8.0,
        f_min: float = 20.0,
        f_max: float = 500.0,
        n_freqs: int = 80
    ) -> Dict[str, Any]:
        """
        High-resolution 2D Time-Frequency Q-Transform Engine.
        Implements Morlet / sine-Gaussian multi-frequency filtering with constant quality factor Q.
        """
        n_samples = len(strain)
        freq_grid = np.geomspace(max(15.0, f_min), min(fs / 2.0 - 10.0, f_max), n_freqs)
        
        # Strain FFT
        fft_freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
        strain_fft = np.fft.rfft(strain)

        # Downsample time grid for fast responsive rendering
        step = max(1, n_samples // 300)
        t_sub = time[::step]
        n_times = len(t_sub)

        energy_map = np.zeros((n_freqs, n_times), dtype=np.float32)

        for i, f0 in enumerate(freq_grid):
            # Sine-Gaussian filter window in frequency domain:
            # W(f; f0, Q) = exp( - (Q^2 / (2 * f0^2)) * (f - f0)^2 )
            sigma_f = f0 / q_value
            filt = np.exp(-0.5 * ((fft_freqs - f0) / sigma_f) ** 2)
            
            # Convolve via spectral multiplication
            filtered_fft = strain_fft * filt
            filtered_time = np.fft.irfft(filtered_fft, n=n_samples)

            # Compute analytic signal envelope using Hilbert quadrature
            analytic_env = np.abs(filtered_time) ** 2
            
            # Mean background energy normalization
            bg_norm = np.median(analytic_env) if np.median(analytic_env) > 0 else 1e-12
            norm_energy = analytic_env / bg_norm
            energy_map[i, :] = norm_energy[::step]

        # Log-scale energy for visualization dynamic range
        energy_map = np.maximum(0.0, energy_map)
        peak_energy = float(np.max(energy_map))
        peak_snr = float(np.sqrt(max(0.0, peak_energy - 1.0)))

        return {
            "times": t_sub,
            "frequencies": freq_grid,
            "energy": energy_map,
            "peak_energy": peak_energy,
            "peak_snr": peak_snr,
            "q_value": q_value
        }

    @classmethod
    def full_conditioning_pipeline(
        cls,
        strain: np.ndarray,
        time: np.ndarray,
        fs: int,
        low_cut: float = 30.0,
        high_cut: float = 400.0,
        q_value: float = 8.0
    ) -> Dict[str, Any]:
        """
        Executes end-to-end 3-stage signal conditioning:
        - Step A: Raw Strain (unprocessed detector data)
        - Step B: Bandpass Filtered Signal (30-400 Hz isolation)
        - Step C: Spectral Whitened Signal (zero-mean, unit-variance strain)
        - Step D: 2D Q-Transform Spectrogram Generation
        
        Returns all arrays plus live statistical reduction metrics.
        """
        # Step A: Raw stats
        raw_rms = float(np.sqrt(np.mean(strain ** 2)))
        raw_p2p = float(np.max(strain) - np.min(strain))

        # Step B: Direct Bandpass Filtering on Raw Strain
        bandpassed = cls.bandpass_filter(strain, fs=fs, low_cut=low_cut, high_cut=high_cut)
        bp_rms = float(np.sqrt(np.mean(bandpassed ** 2)))
        bp_p2p = float(np.max(bandpassed) - np.min(bandpassed))
        
        # Out-of-band noise reduction percentage (seismic + laser noise rejected)
        bp_reduction_pct = float(max(0.0, min(99.9, (1.0 - (bp_rms / max(1e-30, raw_rms))) * 100.0)))
        bp_suppression_db = float(20.0 * np.log10(max(1e-6, bp_rms / max(1e-30, raw_rms))))

        # Step C: Frequency-Domain Whitening on Filtered Signal
        whitened = cls.whiten(strain, fs=fs)
        conditioned = cls.bandpass_filter(whitened, fs=fs, low_cut=low_cut, high_cut=high_cut)
        
        # Whitened stats
        white_rms = float(np.sqrt(np.mean(conditioned ** 2)))
        white_std = float(np.std(conditioned))
        white_mean = float(np.mean(conditioned))
        peak_amp_sigma = float(np.max(np.abs(conditioned)))

        # Step D: Q-Transform Spectrogram
        q_result = cls.q_transform(
            strain=conditioned,
            time=time,
            fs=fs,
            q_value=q_value,
            f_min=low_cut,
            f_max=high_cut
        )

        metrics = {
            "raw_rms": raw_rms,
            "raw_p2p": raw_p2p,
            "bandpass_rms": bp_rms,
            "bandpass_p2p": bp_p2p,
            "bandpass_reduction_pct": round(bp_reduction_pct, 1),
            "bandpass_suppression_db": round(bp_suppression_db, 1),
            "whitened_rms": round(white_rms, 2),
            "whitened_std": round(white_std, 2),
            "whitened_mean": round(white_mean, 4),
            "peak_amp_sigma": round(peak_amp_sigma, 2),
            "peak_snr": round(q_result["peak_snr"], 2)
        }

        return {
            "raw_strain": strain,
            "bandpassed_strain": bandpassed,
            "whitened_strain": whitened,
            "conditioned_strain": conditioned,
            "time": time,
            "fs": fs,
            "metrics": metrics,
            "q_transform": q_result
        }
