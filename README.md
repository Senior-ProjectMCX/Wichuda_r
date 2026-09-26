# Wichuda_r

DCS simulation and evaluation for Senior Project.

## Setup

Activate virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies if needed:

```powershell
pip install -r dcs_simulator\requirements-ui.txt
```

## Run GUI

```powershell
python dcs_simulator\gui.py
```

## Evaluation

### 1. Evaluate the model

Compare Model g2 with Physical Phantom g2 for all experiments.

```powershell
python dcs_simulator\evaluation.py
```

Output:

```text
dcs_simulator/output/evaluation/full_dataset_evaluation.csv
```

The evaluation calculates:

- RMSE for each SDS
- R² for each SDS
- RMSE for each experiment
- Overall RMSE
- Results with and without the `(0,0)` flow condition

### 2. Analyze evaluation results

```powershell
python dcs_simulator\evaluation_analysis.py
```

This analyzes:

- Error for each flow condition
- Lowest and highest error conditions
- Zero-flow conditions
- Representative Physical vs Model curves

Output:

```text
dcs_simulator/output/evaluation/condition_summary.csv
dcs_simulator/output/evaluation/plots/
```

### 3. Analyze Physical Phantom variability

Compare Physical Phantom trials from the same flow condition.

```powershell
python dcs_simulator\physical_variability.py
```

Output:

```text
dcs_simulator/output/evaluation/physical_pairwise_rmse.csv
dcs_simulator/output/evaluation/physical_variability_by_condition.csv
```

This is used to see how much the Physical Phantom measurements vary between trials.

## Evaluation Notes

More details about the evaluation method and current results:

```text
EVALUATION_NOTES.md
```

## Reference Data

Physical Phantom dataset:

```text
dcs_simulator/reference_data/Dataset_phantom-intra40.mat
```

## Current Evaluation Target

Two RMSE targets are currently being considered:

- `RMSE <= 0.03` - based on Physical Phantom trial-to-trial variability
- `RMSE <= 0.10` - proposed practical baseline target

The final threshold has not been selected yet.