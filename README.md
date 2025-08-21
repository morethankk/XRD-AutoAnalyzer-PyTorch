# XRD AutoAnalyzer - PyTorch Implementation

This is a simple PyTorch implementation of the [XRD-AutoAnalyzer](https://github.com/njszym/XRD-AutoAnalyzer) project, designed for automatic identification of phases in X-ray diffraction (XRD) patterns. This implementation retains the data augmentation capabilities of the original project, reimplements the network architecture using PyTorch, and optimizes memory usage during data augmentation.

## Key Features

- **PyTorch Implementation**: Full reimplementation of the neural network architecture using PyTorch
- **Data Augmentation**: Sophisticated data augmentation techniques for XRD spectra including:
  - Peak shifts (strain)
  - Peak intensity changes (texture)
  - Peak broadening (small domain size)
  - Impurity peaks
  - Mixed augmentations
- **Memory Optimization**: Optimized memory usage during data augmentation to prevent out-of-memory errors
- **Monte Carlo Dropout**: Built-in uncertainty estimation through Monte Carlo Dropout

## Architecture Overview

The model is a 1D Convolutional Neural Network with the following structure:

- **6 Convolutional Layers**: With decreasing kernel sizes (35, 30, 25, 20, 15, 10) and ReLU activation
- **Max Pooling**: Applied after each convolutional layer
- **3 Fully Connected Layers**: With sizes dynamically calculated based on input dimensions (default: 3100, 1200, n_phases)
- **Monte Carlo Dropout**: Applied throughout the network for uncertainty estimation
- **Batch Normalization**: Applied to fully connected layers

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd XRD_AutoAnalyzer_PyTorch
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model on your CIF files:

```bash
python train.py --data_dir Novel_Space/All_CIFs --augment_each 50 --epochs 100
```

Key training parameters:
- `--data_dir`: Directory containing CIF files for training
- `--augment_each`: Number of augmented samples generated per phase (default: 50)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 32)
- `--lr`: Initial learning rate (default: 0.001)

The training script will automatically:
1. Create a References directory with copies of CIF files
2. Generate augmented training data
3. Train the model with validation
4. Save the best model based on validation loss and accuracy

### Prediction

To predict phases in XRD spectra:

```bash
python predict.py --spectra_dir Spectra --reference_dir References
```

Key prediction parameters:
- `--model`: Path to trained model file (default: model.pt)
- `--classes`: Path to class information file (default: classes.json)
- `--spectra_dir`: Directory containing experimental XRD spectra
- `--reference_dir`: Directory containing reference CIF files
- `--min_conf`: Minimum confidence threshold for phase identification (default: 0.05)
- `--max_phases`: Maximum number of phases to identify (default: 3)
- `--plot`: Enable plotting of results

## Data Structure

The project expects the following directory structure:

```
XRD_AutoAnalyzer_PyTorch/
├── Novel_Space/
│   └── All_CIFs/          # Training CIF files
├── References/             # Reference CIF files (auto-created)
├── Spectra/                # Experimental spectra for prediction
├── spectrum_generation/    # Data augmentation modules
├── model.py               # Neural network architecture
├── train.py               # Training script
├── predict.py    # Prediction script
├── data_utils.py          # Data processing utilities
└── README.md
```

### Input Data Formats

1. **CIF Files**: Crystal structure files in CIF format for training
2. **Spectra Files**: Experimental XRD data in .xy or .txt format (two columns: 2θ, intensity)

## Limitations

Currently, the model has limited capability for multi-phase identification. The model performs well on single-phase identification but struggles with accurately identifying multiple phases in mixed samples.
