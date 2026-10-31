# Model B: Digital Twin-inspired DCS Phantom

**Senior Project 492 | Computer Engineering, Chiang Mai University (2026)**

This repository contains **Model B**, a Python-based, MCX-driven two-layer phantom simulation for Diffuse Correlation Spectroscopy (DCS). It simulates detected-photon paths and calculates DCS correlation signals for comparison with a physical phantom reference dataset.

Related repositories: [Model A](https://github.com/Senior-ProjectMCX/wachirawit.chai) | [Core: reference data and A/B evaluation](https://github.com/Senior-ProjectMCX/core)

## 1. Repository structure

```text
Wichuda_r/
├── README.md
├── .gitignore
└── dcs_simulator/
    ├── parameters.py              # Model geometry, optical and simulation parameters
    ├── phantom.py                 # Two-layer voxel phantom
    ├── simulation.py              # MCX simulation and photon-data export
    ├── dcs_processor.py           # Photon-path processing and DCS correlation
    ├── flow_conversion.py         # Flow-rate conversion
    ├── reference_loader.py        # Physical phantom reference-data loading
    ├── dataset_generator.py       # Dataset generation
    ├── evaluation.py              # Model evaluation
    ├── evaluation_analysis.py     # Evaluation summaries and plots
    ├── physical_variability.py    # Physical measurement variability analysis
    ├── baseline_model.py          # Supporting baseline analysis
    ├── model_comparison.py        # Supporting baseline comparison
    ├── bfi_estimation.py          # BFI estimation utility
    ├── check_model_b_metrics.py   # Metric verification
    ├── check_my_original_tuning.py # Tuning verification
    ├── human_model.py             # Exploratory synthetic-model utility
    ├── gui.py                     # Internal plotting aid for parameter tuning only
    ├── requirements-ui.txt        # Python dependencies
    ├── Wichuda_r.code-workspace   # VS Code workspace configuration
    ├── reference_data/           # Physical phantom reference dataset
    ├── baseline_data/            # Stored baseline data
    └── output/
        ├── datasets/             # Generated datasets
        ├── evaluation/           # Evaluation metrics and plots
        ├── model_comparison/      # Supporting comparison results
        └── detp_*.npy / .json    # Detected-photon data and metadata
```

## 2. Model parameters

| Parameter | Model B |
|---|---|
| Geometry | 100 × 50 × 40.5 mm; shallow layer 5.5 mm, deep layer 35 mm |
| Source–detector separation (SDS) | 10, 20, 30 mm |
| Launched photons | 10⁷ |
| Optical properties | μa = 0.56 cm⁻¹; μs′ = 4.81 cm⁻¹; g = 0.89; n = 1.37; λ = 785 nm |
| Flow mapping | αD_B = FR × 0.4 × 10⁻⁶ mm²/s |
| Partial pathlength | mm (0.5 mm/voxel) |
| β | Reference g₂(0) − 1 |

See `dcs_simulator/parameters.py` for additional configuration details. The flow coefficient was selected through manual tuning.

## 3. Installation

The simulation requires Python, the dependencies below, and a compatible GPU/MCX environment for new photon simulations. The project was developed using Python 3.13.

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\dcs_simulator\requirements-ui.txt
```

## 4. Running the simulation

From the repository root:

```powershell
# Check model parameters and DCS processing equations
python .\dcs_simulator\parameters.py
python .\dcs_simulator\dcs_processor.py

# Check MCX configuration without launching photons
python .\dcs_simulator\simulation.py --dry-run

# Run a photon simulation (requires a compatible GPU)
python .\dcs_simulator\simulation.py --name trial_01 --nphoton 10000000
```

The simulation exports detected-photon data and metadata to `dcs_simulator/output/`. Choose a new `--name` for each run to avoid overwriting existing results.

## 5. Dataset generation and evaluation

```powershell
python .\dcs_simulator\dataset_generator.py
python .\dcs_simulator\evaluation.py
python .\dcs_simulator\evaluation_analysis.py
```

These scripts use the available reference data and stored simulation outputs; running them may regenerate output files. Additional analysis and verification scripts are available in `dcs_simulator/`.

### Final evaluation summary

The final comparison used **132 nonzero-flow conditions**, **5 trials per condition**, and **3 SDS values**, yielding **1,980 correlation curves**.

| SDS | Model B median R² | Model B pooled RMSE |
|---|---:|---:|
| 1 cm | 0.691 | 0.088 |
| 2 cm | 0.552 | 0.072 |
| 3 cm | 0.516 | 0.060 |

These figures summarize the final project evaluation. The shared [Core repository](https://github.com/Senior-ProjectMCX/core) contains the A/B comparison workflow and reference materials. The model is intended for simulation and research evaluation, not clinical diagnosis.
