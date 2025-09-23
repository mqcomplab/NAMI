# NAMI Chemical Space Visualizer (Readme not ready)

A comprehensive GUI application for molecular clustering and visualization using NAMI.

Refer to the paper: 

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-windows%20%7C%20macOS%20%7C%20linux-lightgrey.svg)

## 🚀 Features

- **Interactive Molecular Clustering**: Cluster molecules using BitBirch algorithm
- **PCA Visualization**: 2D visualization of high-dimensional molecular fingerprints
- **Real-time Exploration**: Click clusters to explore individual molecules
- **Molecular Properties**: Display detailed molecular descriptors and Lipinski properties
- **Save/Load Results**: Persist clustering results for later analysis
- **Customizable Parameters**: Adjust clustering and fingerprint parameters
- **Progress Tracking**: Real-time progress bars for long-running operations

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [File Formats](#file-formats)
- [Parameters](#parameters)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/bitbirch-pca-gui.git
cd bitbirch-pca-gui

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
# Create a new conda environment
conda create -n bitbirch python=3.8
conda activate bitbirch

# Install RDKit via conda (recommended)
conda install -c conda-forge rdkit

# Install other dependencies
pip install pandas numpy matplotlib scikit-learn bitbirch tqdm scipy mplcursors pillow
```

### Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
rdkit-pypi>=2022.3.5
bitbirch>=0.2.0
tqdm>=4.62.0
scipy>=1.7.0
mplcursors>=0.4
Pillow>=8.3.0
```

## 🚀 Quick Start

1. **Launch the application:**
   ```bash
   python main.py
   ```

2. **Load your data:**
   - Click "Load SMILES CSV"
   - Select a CSV file containing SMILES strings

3. **Run clustering:**
   - Adjust parameters if needed
   - Click "Process & Cluster"

4. **Explore results:**
   - Click on clusters in the overview plot
   - Hover over molecules for detailed information

## 📖 Usage

### Main Interface

The application consists of several panels:

- **Controls Panel**: Load data, set parameters, and control processing
- **Data Information**: Shows dataset statistics and sample molecules
- **Visualization**: Interactive PCA plots of clusters and molecules
- **Clustering Results**: Detailed clustering statistics and parameters
- **Molecule Information**: Chemical structure and properties
- **Additional Details**: Extended molecular descriptors

### Workflow

1. **Data Loading**
   ```
   Load SMILES CSV → Validate Data → Display Info
   ```

2. **Processing**
   ```
   Generate Fingerprints → BitBirch Clustering → PCA Reduction → Visualization
   ```

3. **Exploration**
   ```
   Overview Plot → Click Clusters → Detail View → Hover Molecules → Properties
   ```

### Navigation

- **🔍 Zoom**: Mouse scroll wheel
- **📱 Pan**: Click and drag
- **🎯 Cluster Selection**: Click on cluster centers
- **ℹ️ Molecule Details**: Hover over points in detail view
- **🔄 Reset View**: Reset Zoom button
- **⬅️ Navigation**: Back to Overview button

## 📄 File Formats

### Input CSV Format

**Option 1: Header with SMILES column**
```csv
SMILES,Name
CCO,ethanol
C1=CC=CC=C1,benzene
CC(=O)O,acetic acid
CC(C)O,isopropanol
```

**Option 2: Space-separated (no header)**
```
CCO ethanol
C1=CC=CC=C1 benzene
CC(=O)O acetic_acid
CC(C)O isopropanol
```

**Option 3: SMILES only**
```csv
SMILES
CCO
C1=CC=CC=C1
CC(=O)O
CC(C)O
```

### Output Format

Results are saved as NumPy `.npy` files containing:
- SMILES strings
- Cluster assignments
- Fingerprint matrix
- PCA coordinates
- Parameters used

## ⚙️ Parameters

### BitBirch Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **BB Threshold** | 0.65 | Distance threshold for clustering |
| **Branching Factor** | 50 | Maximum children per node |

### Fingerprint Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **FP Radius** | 2 | Morgan fingerprint radius |
| **FP Bits** | 2048 | Fingerprint bit vector size |

### Display Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Min Large Cluster** | 10 | Minimum cluster size for overview |
| **Max Large Cluster** | 1000 | Maximum cluster size for overview |

### Parameter Guidelines

- **Lower threshold** → More, smaller clusters
- **Higher threshold** → Fewer, larger clusters
- **Higher radius** → More specific fingerprints
- **More bits** → Higher resolution fingerprints

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'rdkit'`

**Solution**:
```bash
# Try conda installation
conda install -c conda-forge rdkit

# Or pip
pip install rdkit-pypi
```

#### 2. GUI Issues

**Problem**: GUI doesn't appear or crashes

**Solutions**:
```bash
# Linux/Ubuntu
sudo apt-get install python3-tk

# Verify matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"
```

#### 3. Memory Issues

**Problem**: Out of memory with large datasets

**Solutions**:
- Reduce fingerprint bits (e.g., 1024 instead of 2048)
- Process smaller subsets of data
- Increase system virtual memory

#### 4. Slow Performance

**Problem**: Clustering takes too long

**Solutions**:
- Increase BitBirch threshold
- Reduce branching factor
- Use smaller fingerprint radius

### Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "No file loaded" | CSV file not selected | Click "Load SMILES CSV" |
| "Invalid SMILES" | Malformed molecular structures | Check SMILES syntax |
| "No clusters found" | Threshold too high | Lower BB threshold |
| "Memory error" | Dataset too large | Reduce data size or parameters |

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests if applicable**
5. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
6. **Push to your branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/bitbirch-pca-gui.git
cd bitbirch-pca-gui

# Install in development mode
pip install -e .

# Run tests (if available)
python -m pytest tests/
```

## 📊 Example Results

### Sample Dataset Results
- **Molecules**: 1,000 compounds
- **Clusters Found**: 23 clusters
- **Largest Cluster**: 156 molecules
- **Processing Time**: ~30 seconds

### Typical Use Cases
- **Drug Discovery**: Cluster compound libraries
- **Chemical Space Analysis**: Visualize molecular diversity
- **Lead Optimization**: Group similar structures
- **Library Design**: Identify structural gaps

## 📝 Changelog

### Version 1.0.0
- Initial release
- BitBirch clustering implementation
- PCA visualization
- Interactive molecule exploration
- Save/load functionality

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [RDKit](https://www.rdkit.org/) for molecular informatics
- [BitBirch](https://github.com/example/bitbirch) for clustering algorithm
- [scikit-learn](https://scikit-learn.org/) for PCA implementation
- [matplotlib](https://matplotlib.org/) for visualization

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/bitbirch-pca-gui/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/bitbirch-pca-gui/discussions)
- **Email**: your.email@example.com

---

**Made with ❤️ for the molecular modeling community**
