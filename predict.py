import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.signal import savgol_filter, find_peaks
from model import XRDNet
from data_utils import simulate_pattern_from_cif


def load_model(model_path, class_info, input_length, device, dropout_rate=0.0):
    num_classes = len(class_info["class_names"])
    temp_model = XRDNet(n_phases=num_classes, dropout_rate=dropout_rate)
    flattened_size = temp_model.get_num_features(input_length)
    print(f"Loading model with flattened feature size: {flattened_size}")
    
    model = XRDNet(n_phases=num_classes, dropout_rate=dropout_rate, n_dense=[min(3100, flattened_size), min(1200, flattened_size//2)]).to(device)
    del temp_model 
    
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def baseline_als(y, lam=10**6, p=0.01, niter=10):
    """
    Asymmetric Least Squares baseline correction.
    """
    L = len(y)
    D = np.diff(np.eye(L), 2)
    w = np.ones(L)
    for i in range(niter):
        W = np.diag(w)
        Z = W + lam * D.dot(D.T)
        z = np.linalg.solve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def enhanced_background_removal(spectrum):
    """
    Enhanced background removal using ALS.
    """
    # Apply Asymmetric Least Squares for baseline estimation
    baseline = baseline_als(spectrum, lam=10**6, p=0.01, niter=10)
    
    # Subtract baseline
    corrected = spectrum - baseline
    
    # Ensure non-negative values
    corrected = np.maximum(corrected, 0)
    
    return corrected, baseline

def smooth_spectrum(spectrum, window_length=15, polyorder=3):
    """
    Smooth spectrum using Savitzky-Golay filter.
    """
    if window_length > len(spectrum):
        window_length = len(spectrum) // 2 * 2 + 1  # Ensure odd window length
        if window_length < 5:
            return spectrum  # Too short to smooth
    
    if polyorder >= window_length:
        polyorder = window_length - 1
    
    smoothed = savgol_filter(spectrum, window_length, polyorder)
    return smoothed

def load_spectrum(file_path, min_angle, max_angle, step):
    data = np.loadtxt(file_path)
    two_theta = data[:, 0]
    intensity = data[:, 1]
    grid = np.arange(min_angle, max_angle + step, step)
    interp_intensity = np.interp(grid, two_theta, intensity, left=0, right=0)
    
    # Apply smoothing
    interp_intensity = smooth_spectrum(interp_intensity)
    
    # Enhanced background removal
    interp_intensity, _ = enhanced_background_removal(interp_intensity)
    
    if interp_intensity.max() > 0:
        interp_intensity /= interp_intensity.max()
    return grid, interp_intensity

def subtract_reference_peaks(spectrum, reference, weight):
    """
    Subtract the reference spectral lines (scaled by weight) from the spectrum, avoiding negative values.
    """
    reference_scaled = reference * weight
    subtracted = spectrum - reference_scaled
    return np.maximum(subtracted, 0)  # Ensure non-negative values

def find_matching_cif(reference_dir, phase_name):
    for f in os.listdir(reference_dir):
        if not f.lower().endswith(".cif"):
            continue
        if phase_name.split("_")[0] in f:
            return os.path.join(reference_dir, f)
    return None

def main():
    parser = argparse.ArgumentParser(description="Interactive XRD Phase Prediction")
    parser.add_argument("--model", type=str, default="model.pt")
    parser.add_argument("--classes", type=str, default="classes.json")
    parser.add_argument("--spectra_dir", type=str, default="Spectra")
    parser.add_argument("--reference_dir", type=str, default="References")
    parser.add_argument("--min_conf", type=float, default=0.05)  
    parser.add_argument("--max_phases", type=int, default=3)  
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save_plot", action="store_true")
    args = parser.parse_args()

    # Load class info
    with open(args.classes, "r") as f:
        class_info = json.load(f)

    class_names = class_info["class_names"]
    min_angle = class_info["min_angle"]
    max_angle = class_info["max_angle"]
    step = class_info["step"]
    input_length = int((max_angle - min_angle) / step + 1)
    grid = np.linspace(min_angle, max_angle, input_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, class_info, input_length, device)

    files = [f for f in os.listdir(args.spectra_dir) if f.lower().endswith((".txt", ".xy", ".csv"))]
    files.sort()

    for fname in files:
        fpath = os.path.join(args.spectra_dir, fname)
        angles, spectrum = load_spectrum(fpath, min_angle, max_angle, step)
        orig_spectrum = spectrum.copy()
        remaining = spectrum.copy()

        pred_phases = []
        pred_conf = []

        print(f"\n📂 文件: {fname}")

        for step_i in range(args.max_phases):
            x = torch.tensor(remaining, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
            
            best_idx = int(np.argmax(probs))
            conf = probs[best_idx]
            phase = class_names[best_idx]

            # Termination condition 1: Low confidence
            if conf < args.min_conf:
                print(f"Step {step_i+1}: Confidence ({conf:.3f}) below threshold({args.min_conf})")
                break
            
            # Termination condition 2: Duplicate phases
            if phase in pred_phases:
                print(f"Step {step_i+1}: Duplicate phases detected {phase}")
                break

            pred_phases.append(phase)
            pred_conf.append(conf)

            print(f"Step {step_i+1}: phases detected {phase} (Confidence: {conf*100:.2f}%)")

            # Deduct reference peaks
            cif_path = find_matching_cif(args.reference_dir, phase)
            if cif_path:
                _, ref_profile = simulate_pattern_from_cif(cif_path, min_angle, max_angle, step)
                if ref_profile is not None and ref_profile.max() > 0:
                    ref_profile /= ref_profile.max()
                    weight = conf  # 或者 conf * remaining.max()
                    remaining = subtract_reference_peaks(remaining, ref_profile, weight)

            # Termination condition 3: The spectral lines are nearly residue-free
            if remaining.max() < 0.02:  # Lowered threshold
                print(f"Step{step_i+1}: Residual signal is too low (< 0.02)")
                break

        # Print results
        if pred_phases:
            print(f"\n🎯 Prediction results:")
            for p, c in zip(pred_phases, pred_conf):
                print(f"  🔍 {p} (Confidence: {c*100:.2f}%)")
        else:
            print("⚠️ No highly confident phases were identified")

if __name__ == "__main__":
    main()
