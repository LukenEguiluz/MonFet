import numpy as np
from scipy import signal
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

# Function to simulate ECG signals (replace this with actual data acquisition)
def simulate_ecg_signals(duration, sampling_rate):
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    maternal_ecg = signal.sawtooth(2 * np.pi * 1.0 * t)  # Simulated maternal ECG
    fetal_ecg = signal.sawtooth(2 * np.pi * 2.5 * t)     # Simulated fetal ECG (higher frequency)
    noise = 0.1 * np.random.normal(size=t.size)          # Simulated noise
    return maternal_ecg, fetal_ecg, noise

# Function to mix signals based on electrode positions
def mix_signals(maternal_ecg, fetal_ecg, noise):
    mixed_signals = []
    # Electrode 1: Right chest (strong maternal ECG)
    mixed_signals.append(0.9 * maternal_ecg + 0.1 * fetal_ecg + noise)
    # Electrode 2: Left leg in iliac crest (moderate maternal, some fetal)
    mixed_signals.append(0.6 * maternal_ecg + 0.4 * fetal_ecg + noise)
    # Electrode 3: Precordial V5 (strong maternal ECG)
    mixed_signals.append(0.8 * maternal_ecg + 0.2 * fetal_ecg + noise)
    # Electrode 4: Under sternum (moderate maternal ECG)
    mixed_signals.append(0.7 * maternal_ecg + 0.3 * fetal_ecg + noise)
    # Electrodes 5, 6, 7: Around baby's heart (strong fetal ECG)
    for _ in range(3):
        mixed_signals.append(0.2 * maternal_ecg + 0.8 * fetal_ecg + noise)
    # Electrodes 8, 9, 10: Around baby's brain (strong fetal ECG)
    for _ in range(3):
        mixed_signals.append(0.1 * maternal_ecg + 0.9 * fetal_ecg + noise)
    return np.array(mixed_signals)

# Function to apply adaptive filtering
def adaptive_filter(reference_signal, input_signal, filter_order, step_size):
    n_samples = len(input_signal)
    weights = np.zeros(filter_order)
    output_signal = np.zeros(n_samples)
    for i in range(filter_order, n_samples):
        x = input_signal[i - filter_order:i]
        output_signal[i] = np.dot(weights, x)
        error = reference_signal[i] - output_signal[i]
        weights += step_size * error * x
    return output_signal

# Function to separate signals using ICA
def separate_signals_ica(mixed_signals):
    ica = FastICA(n_components=10, random_state=0)
    ica_sources = ica.fit_transform(mixed_signals.T)
    return ica_sources.T

# Main program
def main():
    # Simulate ECG signals
    duration = 5  # seconds
    sampling_rate = 1000  # Hz
    maternal_ecg, fetal_ecg, noise = simulate_ecg_signals(duration, sampling_rate)

    # Mix signals based on electrode positions
    mixed_signals = mix_signals(maternal_ecg, fetal_ecg, noise)

    # Apply adaptive filtering to reduce maternal ECG
    filtered_signals = []
    for i in range(mixed_signals.shape[0]):
        if i in [0, 2]:  # Electrodes with strong maternal ECG (right chest, precordial V5)
            reference_signal = mixed_signals[i]
        else:
            reference_signal = mixed_signals[0]  # Use right chest as reference
        filtered_signal = adaptive_filter(reference_signal, mixed_signals[i], filter_order=10, step_size=0.01)
        filtered_signals.append(filtered_signal)
    filtered_signals = np.array(filtered_signals)

    # Separate signals using ICA
    ica_sources = separate_signals_ica(filtered_signals)

    # Plot results
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.title("Mixed Signals")
    for i in range(mixed_signals.shape[0]):
        plt.plot(mixed_signals[i], label=f"Electrode {i+1}")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.title("Filtered Signals (After Adaptive Filtering)")
    for i in range(filtered_signals.shape[0]):
        plt.plot(filtered_signals[i], label=f"Electrode {i+1}")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.title("Separated Signals (After ICA)")
    for i in range(ica_sources.shape[0]):
        plt.plot(ica_sources[i], label=f"Source {i+1}")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()