# Model B: Digital Twin-inspired DCS Phantom (Wichuda_r)

Senior Project 492, Computer Engineering, Chiang Mai University (2026). This repository contains **Model B**, a Python/MCX two-layer phantom simulator for Diffuse Correlation Spectroscopy (DCS). It generates detected-photon histories and simulated `g1(τ)`/`g2(τ)` signals and provides tools for comparison with the physical phantom reference. Model B uses a literature-informed forward model with a manually selected flow coefficient; it is **not** clinically validated and does not use patient data.

Related repositories: [Model A](https://github.com/Senior-ProjectMCX/wachirawit.chai) · [Core / reference and shared evaluation](https://github.com/Senior-ProjectMCX/core). This repository is the Model B implementation, not the authoritative location for the final cross-model A/B comparison.

## 1. Repository structure and file roles

```text
Wichuda_r/
├── README.md                         Main project documentation
├── .gitignore                        Git exclusions
└── dcs_simulator/
    ├── parameters.py                 Geometry, optical, flow and delay-time constants
    ├── phantom.py                    Two-layer voxel phantom and MCX optical properties
    ├── simulation.py                 MCX simulation CLI, photon-history export and dry-run
    ├── dcs_processor.py              DPM extraction, voxel-to-mm pathlength, g1/g2 calculation
    ├── flow_conversion.py            Flow to alpha*Db conversion
    ├── reference_loader.py           Physical phantom MAT reference loading
    ├── baseline_model.py             Baseline/reference-model utilities
    ├── dataset_generator.py          Synthetic dataset generation
    ├── human_model.py                Synthetic human-model input representation
    ├── gui.py                        Optional internal plotting/tuning aid (not a project deliverable)
    ├── evaluation.py                 Model-versus-physical evaluation
    ├── evaluation_analysis.py        Evaluation summaries and plots
    ├── physical_variability.py       Physical trial-to-trial variability analysis
    ├── model_comparison.py           Earlier baseline-model comparison (see note below)
    ├── bfi_estimation.py             Auxiliary effective BFI fitting/comparison
    ├── check_model_b_metrics.py      Model B metric recalculation / verification
    ├── check_my_original_tuning.py   Checks alternate R² definitions from recalculated CSV
    ├── requirements-ui.txt           Pinned Python dependencies
    ├── Wichuda_r.code-workspace      Optional VS Code workspace settings
    ├── reference_data/              Physical phantom MAT dataset
    ├── baseline_data/               Stored photon pathlength baseline (.npz)
    └── output/
        ├── detp_core_v1.npy          Stored detected-photon histories used by GUI
        ├── detp_human_v1.npy         Stored human-model photon histories
        ├── detp_test_01.npy/.json    Test photon histories and simulation metadata
        ├── datasets/                Generated/reference dataset CSVs
        ├── evaluation/              Evaluation CSVs and illustrative plots
        └── model_comparison/        Historical comparison CSVs and heatmaps
```

**File status:** `model_comparison.py` explicitly describes its baseline comparator as *temporary/older*; its saved comparisons must not be presented as the final Model A vs Model B results. `check_model_b_metrics.py` and `check_my_original_tuning.py` are verification/analysis scripts, not model-training or optimization routines. Stored `output/` artifacts are retained for reproducibility and illustration; regenerating them can overwrite or change results. The `.code-workspace` file is optional. `gui.py` was retained solely as an internal convenience for quickly inspecting plots while manually tuning parameters; interface design and GUI usability were not objectives or evaluated deliverables of this project. The `human_model.py` and related output are synthetic exploratory functionality, not human-subject validation.

## 2. Model configuration

The following summarizes the main Model B settings used in the final implementation (see `dcs_simulator/parameters.py` and `simulation.py` for the complete configuration).

| Parameter | Model B |
|---|---|
| Geometry | 100 × 50 × 40.5 mm (shallow 5.5 mm; deep 35 mm) |
| Detector | SDS 10, 20, 30 mm |
| Launched photons | 10⁷ |
| Optical properties | μa = 0.56 cm⁻¹; μs′ = 4.81 cm⁻¹; g = 0.89; n = 1.37; λ = 785 nm |
| Flow → αD_B | `αD_B = FR × 0.4 × 10⁻⁶ mm²/s` |
| Partial pathlength unit | mm (converted from voxel units using 0.5 mm/voxel) |
| β | Reference `g₂(0) − 1` |

The flow coefficient was selected during earlier manual tuning and was **not re-optimized** after the partial-pathlength unit correction. The flow mapping is a model assumption, not a validated physiological conversion.

## 3. Installation

Requirements: Python compatible with the pinned packages (developed with Python 3.13), a virtual environment, and a working PMCX/MCX installation with compatible GPU/CUDA support for actual photon simulations. Viewing stored results and running some checks does not require a new GPU simulation.

From the repository root in **PowerShell**:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\dcs_simulator\requirements-ui.txt
```

The repository includes `dcs_simulator/reference_data/Dataset_phantom-intra40.mat` and example/stored `output/` files. Check local data paths and dependencies before regenerating results.

## 4. Running Model B (command line)

Run from the **repository root**:

```powershell
# Check parameters and equations (no MCX simulation)
python .\dcs_simulator\parameters.py
python .\dcs_simulator\dcs_processor.py

# Validate MCX setup without running photons
python .\dcs_simulator\simulation.py --dry-run

# Run MCX and save a new photon-history file (requires compatible GPU)
python .\dcs_simulator\simulation.py --name trial_01 --nphoton 10000000
```

The simulation CLI saves `output/detp_<name>.npy` and corresponding JSON metadata. Use a **new** name to avoid `FileExistsError`. Add `--activate` only when you intentionally want to replace the default stored `output/detp_core_v1.npy` used by the internal plotting utility; the script first creates a timestamped backup. Additional options: `--gpu-id` and `--radius-voxels`. Do not use `--activate` casually when reproducing a saved result.

The optional `gui.py` was created only to make plot inspection easier during manual parameter tuning. It is **not** the main execution workflow, a user-facing application, or a GUI/design deliverable. The project focuses on the MCX-based DCS model and its quantitative evaluation.

## 5. Dataset generation and evaluation

The following scripts can generate or overwrite output files. Run them only when their required reference data and photon histories are present and you intend to regenerate results.

```powershell
python .\dcs_simulator\dataset_generator.py
python .\dcs_simulator\evaluation.py
python .\dcs_simulator\evaluation_analysis.py
python .\dcs_simulator\physical_variability.py
```

Key saved outputs include `output/evaluation/full_dataset_evaluation.csv`, `condition_summary.csv`, `physical_pairwise_rmse.csv`, `physical_variability_by_condition.csv`, and figures in `output/evaluation/plots/`.

Additional verification scripts:

```powershell
python .\dcs_simulator\check_model_b_metrics.py
python .\dcs_simulator\check_my_original_tuning.py
```

Run the first script before the second because `check_my_original_tuning.py` reads `output/evaluation/model_b_recheck.csv`. These checks distinguish squared Pearson correlation (`R²_corr`) from the coefficient of determination (`R²_SS = 1 − SSE/SST`); do not interchange their results.

## 6. Final project evaluation context

In the final project comparison, 132 nonzero-flow conditions were sampled at five physical trials per condition (660 trials), with three SDS curves per trial (1,980 curves). The reported Model B results were:

| SDS | Median R² (final comparison) | Pooled RMSE |
|---|---:|---:|
| 1 cm | 0.691 | 0.088 |
| 2 cm | 0.552 | 0.072 |
| 3 cm | 0.516 | 0.060 |

These are **final report comparison figures**, not a claim that running the historical `evaluation.py` script or `model_comparison.py` will reproduce the same sampling/metric protocol. The shared Core repository contains the cross-model evaluation context; consult the final report for full methodology and Model A comparison. In the final comparison, Model B used 10⁷ simulated photons. These metrics do not establish clinical accuracy.

## 7. Reference and scope notes

The reference dataset is `dcs_simulator/reference_data/Dataset_phantom-intra40.mat`. Model B compares generated DCS signals with laboratory physical phantom measurements, not patient data. The synthetic human-model features are exploratory and should not be interpreted as validated human physiology. Results and scripts from early exploratory baselines are retained in the repository for traceability but are not interchangeable with the final A/B evaluation.

The prior `EVALUATION_NOTES.md` described an earlier baseline with `flow × 1e-6` and exploratory RMSE thresholds. Those older assumptions are **not** the current Model B configuration and are intentionally not reproduced as current results here. This README replaces that notes file as the repository's primary documentation.
