from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

from gui import run_validation


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

REFERENCE_FILE = (
    BASE_DIR
    / "reference_data"
    / "Dataset_phantom-intra40.mat"
)

EVALUATION_DIR = (
    BASE_DIR
    / "output"
    / "evaluation"
)

RESULT_CSV = (
    EVALUATION_DIR
    / "full_dataset_evaluation.csv"
)

CONDITION_CSV = (
    EVALUATION_DIR
    / "condition_summary.csv"
)

PLOT_DIR = (
    EVALUATION_DIR
    / "plots"
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

# Representative conditions selected from full-dataset results:
#
# (1, 1)  = low-error condition
# (5, 20) = previously inspected condition
# (0, 1)  = high-error non-(0,0) condition

SELECTED_CONDITIONS = [
    (1.0, 1.0, "low_error"),
    (5.0, 20.0, "representative"),
    (0.0, 1.0, "high_error"),
]


# ============================================================
# METRICS
# ============================================================

def calculate_overall_rmse(values):
    """
    Calculate aggregated RMSE from experiment-level RMSE values.

    Overall RMSE =
        sqrt(mean(RMSE_i^2))
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return np.nan

    return float(
        np.sqrt(
            np.mean(
                values ** 2
            )
        )
    )


def calculate_curve_metrics(
    reference_g2,
    predicted_g2,
):
    """
    Calculate RMSE and Pearson r^2 for one physical/model curve pair.
    """

    reference_g2 = np.asarray(
        reference_g2,
        dtype=float,
    )

    predicted_g2 = np.asarray(
        predicted_g2,
        dtype=float,
    )

    if reference_g2.shape != predicted_g2.shape:
        raise ValueError(
            f"Curve shape mismatch: "
            f"reference={reference_g2.shape}, "
            f"model={predicted_g2.shape}"
        )

    error = (
        predicted_g2
        - reference_g2
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    # R² definition used in the evaluation:
    # squared Pearson correlation coefficient.
    #
    # If either curve is constant, Pearson correlation
    # is undefined, so return NaN.

    if (
        np.std(reference_g2) == 0
        or np.std(predicted_g2) == 0
    ):
        r_squared = np.nan

    else:
        correlation = np.corrcoef(
            reference_g2,
            predicted_g2,
        )[0, 1]

        r_squared = float(
            correlation ** 2
        )

    return (
        rmse,
        r_squared,
    )


# ============================================================
# CONDITION-LEVEL ANALYSIS
# ============================================================

def create_condition_summary(df):

    print()
    print("=" * 80)
    print("CONDITION-LEVEL ANALYSIS")
    print("=" * 80)

    rows = []

    grouped = df.groupby(
        [
            "commanded_shallow",
            "commanded_deep",
        ]
    )

    for (
        shallow,
        deep,
    ), group in grouped:

        experiment_rmse = (
            group[
                "experiment_rmse"
            ].to_numpy(
                dtype=float
            )
        )

        row = {
            "commanded_shallow":
                shallow,

            "commanded_deep":
                deep,

            "number_of_trials":
                len(group),

            "mean_experiment_rmse":
                float(
                    np.mean(
                        experiment_rmse
                    )
                ),

            "overall_rmse":
                calculate_overall_rmse(
                    experiment_rmse
                ),

            "mean_sds1_rmse":
                group[
                    "sds1_rmse"
                ].mean(),

            "mean_sds2_rmse":
                group[
                    "sds2_rmse"
                ].mean(),

            "mean_sds3_rmse":
                group[
                    "sds3_rmse"
                ].mean(),

            "mean_sds1_r2":
                group[
                    "sds1_r2"
                ].mean(),

            "mean_sds2_r2":
                group[
                    "sds2_r2"
                ].mean(),

            "mean_sds3_r2":
                group[
                    "sds3_r2"
                ].mean(),
        }

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    )

    summary = summary.sort_values(
        "overall_rmse"
    ).reset_index(
        drop=True
    )

    summary.to_csv(
        CONDITION_CSV,
        index=False,
    )

    print(
        f"Number of unique conditions: "
        f"{len(summary)}"
    )

    print()

    print(
        "Lowest-error conditions:"
    )

    print(
        summary[
            [
                "commanded_shallow",
                "commanded_deep",
                "number_of_trials",
                "overall_rmse",
            ]
        ].head(
            10
        ).to_string(
            index=False
        )
    )

    print()

    zero_zero_mask = (
        np.isclose(
            summary[
                "commanded_shallow"
            ],
            0.0,
        )
        &
        np.isclose(
            summary[
                "commanded_deep"
            ],
            0.0,
        )
    )

    non_zero_zero = summary[
        ~zero_zero_mask
    ]

    print(
        "Highest-error conditions "
        "(excluding 0,0):"
    )

    print(
        non_zero_zero[
            [
                "commanded_shallow",
                "commanded_deep",
                "number_of_trials",
                "overall_rmse",
            ]
        ].sort_values(
            "overall_rmse",
            ascending=False,
        ).head(
            10
        ).to_string(
            index=False
        )
    )

    print()

    print(
        "Condition summary saved to:"
    )

    print(
        CONDITION_CSV
    )

    return summary


# ============================================================
# ZERO-FLOW ANALYSIS
# ============================================================

def analyze_zero_flow_groups(df):

    print()
    print("=" * 80)
    print("ZERO-FLOW GROUP ANALYSIS")
    print("=" * 80)

    shallow = df[
        "commanded_shallow"
    ].to_numpy(
        dtype=float
    )

    deep = df[
        "commanded_deep"
    ].to_numpy(
        dtype=float
    )

    masks = {
        "(0,0)": (
            np.isclose(
                shallow,
                0.0,
            )
            &
            np.isclose(
                deep,
                0.0,
            )
        ),

        "Shallow=0, Deep>0": (
            np.isclose(
                shallow,
                0.0,
            )
            &
            (
                deep > 0
            )
        ),

        "Shallow>0, Deep=0": (
            (
                shallow > 0
            )
            &
            np.isclose(
                deep,
                0.0,
            )
        ),

        "Shallow>0, Deep>0": (
            (
                shallow > 0
            )
            &
            (
                deep > 0
            )
        ),
    }

    for name, mask in masks.items():

        group = df.loc[
            mask
        ]

        if len(group) == 0:
            continue

        overall_rmse = (
            calculate_overall_rmse(
                group[
                    "experiment_rmse"
                ].to_numpy(
                    dtype=float
                )
            )
        )

        print(
            f"{name:22s} "
            f"| n={len(group):4d} "
            f"| Overall RMSE="
            f"{overall_rmse:.8f}"
        )


# ============================================================
# LOAD PHYSICAL CURVES
# ============================================================

def load_physical_condition(
    dataset,
    gtau,
    shallow_flow,
    deep_flow,
):
    """
    Load every raw physical trial for one commanded
    shallow/deep condition.

    Dataset_set_intra40:
        row 0 = commanded shallow
        row 1 = commanded deep

    ave_gtau_intra40 shape:
        (10, 501, 1540)

    For this project:
        index 0 = SDS 1 cm
        index 1 = SDS 2 cm
        index 2 = SDS 3 cm

    Final curve array for each SDS:
        (number_of_trials, 501)
    """

    mask = (
        np.isclose(
            dataset[
                0,
                :,
            ],
            shallow_flow,
        )
        &
        np.isclose(
            dataset[
                1,
                :,
            ],
            deep_flow,
        )
    )

    columns = np.where(
        mask
    )[0]

    if len(columns) == 0:
        raise ValueError(
            f"No physical data found for "
            f"Shallow={shallow_flow}, "
            f"Deep={deep_flow}"
        )

    curves = {}

    for sds_cm in (
        1,
        2,
        3,
    ):

        selected = gtau[
            sds_cm - 1,
            :,
            columns,
        ]

        selected = np.asarray(
            selected,
            dtype=float,
        )

        # Depending on NumPy advanced indexing,
        # selected may be either:
        #
        # (number_of_trials, 501)
        #
        # or
        #
        # (501, number_of_trials)
        #
        # Force the final format to:
        # (number_of_trials, 501)

        if selected.shape == (
            len(columns),
            501,
        ):
            curves[
                sds_cm
            ] = selected

        elif selected.shape == (
            501,
            len(columns),
        ):
            curves[
                sds_cm
            ] = selected.T

        else:
            raise ValueError(
                f"Unexpected g2 shape "
                f"for SDS {sds_cm}: "
                f"{selected.shape}. "
                f"Expected "
                f"({len(columns)}, 501) "
                f"or "
                f"(501, {len(columns)})."
            )

    return (
        columns,
        curves,
    )


# ============================================================
# PLOT ONE CONDITION
# ============================================================

def plot_condition(
    dataset,
    gtau,
    shallow_flow,
    deep_flow,
    label,
):

    columns, physical = (
        load_physical_condition(
            dataset=
                dataset,

            gtau=
                gtau,

            shallow_flow=
                shallow_flow,

            deep_flow=
                deep_flow,
        )
    )

    # --------------------------------------------------------
    # Generate model only once for this condition
    # --------------------------------------------------------

    model = run_validation(
        shallow_flow=
            shallow_flow,

        deep_flow=
            deep_flow,
    )

    # --------------------------------------------------------
    # Delay time
    #
    # 501 samples:
    # 0, 1, 2, ..., 500 microseconds
    # --------------------------------------------------------

    tau_us = np.arange(
        501,
        dtype=float,
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(
            16,
            5,
        ),
        sharex=True,
    )

    # Used later to calculate one condition-level
    # RMSE from every physical trial and every SDS.

    all_squared_errors = []

    # ========================================================
    # SDS 1 / 2 / 3
    # ========================================================

    for index, sds_cm in enumerate(
        (
            1,
            2,
            3,
        )
    ):

        axis = axes[
            index
        ]

        physical_curves = np.asarray(
            physical[
                sds_cm
            ],
            dtype=float,
        )

        model_curve = np.asarray(
            model[
                "results"
            ][
                sds_cm
            ][
                "generated_g2"
            ],
            dtype=float,
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if model_curve.shape != (
            501,
        ):
            raise ValueError(
                f"Unexpected model curve shape "
                f"for SDS {sds_cm}: "
                f"{model_curve.shape}"
            )

        # ----------------------------------------------------
        # Plot every RAW physical trial
        # ----------------------------------------------------

        for trial_index, curve in enumerate(
            physical_curves
        ):

            if curve.shape != (
                501,
            ):
                raise ValueError(
                    f"Unexpected physical curve "
                    f"shape for SDS {sds_cm}, "
                    f"trial {trial_index + 1}: "
                    f"{curve.shape}"
                )

            physical_label = (
                "Physical trials"
                if trial_index == 0
                else None
            )

            axis.plot(
                tau_us,
                curve,
                alpha=0.35,
                linewidth=1.0,
                label=physical_label,
            )

        # ----------------------------------------------------
        # Plot model
        # ----------------------------------------------------

        axis.plot(
            tau_us,
            model_curve,
            linestyle="--",
            linewidth=2.0,
            label="Model",
        )

        # ----------------------------------------------------
        # Calculate metrics PER RAW TRIAL
        # ----------------------------------------------------

        trial_rmse = []
        trial_r2 = []

        for curve in physical_curves:

            rmse, r_squared = (
                calculate_curve_metrics(
                    reference_g2=
                        curve,

                    predicted_g2=
                        model_curve,
                )
            )

            trial_rmse.append(
                rmse
            )

            trial_r2.append(
                r_squared
            )

            error = (
                model_curve
                - curve
            )

            all_squared_errors.extend(
                error ** 2
            )

        # ----------------------------------------------------
        # Summary for this SDS
        # ----------------------------------------------------

        mean_rmse = float(
            np.mean(
                trial_rmse
            )
        )

        overall_sds_rmse = (
            calculate_overall_rmse(
                trial_rmse
            )
        )

        valid_r2 = np.asarray(
            trial_r2,
            dtype=float,
        )

        valid_r2 = valid_r2[
            np.isfinite(
                valid_r2
            )
        ]

        if len(valid_r2) > 0:

            mean_r2 = float(
                np.mean(
                    valid_r2
                )
            )

            r2_text = (
                f"{mean_r2:.3f}"
            )

        else:
            r2_text = "N/A"

        # ----------------------------------------------------
        # Plot labels
        # ----------------------------------------------------

        axis.set_title(
            f"SDS {sds_cm} cm\n"
            f"RMSE={overall_sds_rmse:.4f}, "
            f"Mean R²={r2_text}"
        )

        axis.set_xlabel(
            "Delay time (µs)"
        )

        axis.grid(
            alpha=0.25
        )

        axis.legend()

    # --------------------------------------------------------
    # Condition-level RMSE
    #
    # This pools the pointwise squared errors from:
    #
    # every trial
    # × every SDS
    # × all 501 delay-time points
    # --------------------------------------------------------

    all_squared_errors = np.asarray(
        all_squared_errors,
        dtype=float,
    )

    condition_rmse = float(
        np.sqrt(
            np.mean(
                all_squared_errors
            )
        )
    )

    axes[0].set_ylabel(
        "g₂"
    )

    readable_label = (
        label
        .replace(
            "_",
            " ",
        )
        .title()
    )

    figure.suptitle(
        (
            f"{readable_label} Condition\n"
            f"Shallow={shallow_flow:g}, "
            f"Deep={deep_flow:g} "
            f"| Physical trials={len(columns)} "
            f"| Overall RMSE={condition_rmse:.4f}"
        ),
        fontsize=14,
    )

    figure.tight_layout()

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    filename = (
        f"{label}"
        f"_shallow_{shallow_flow:g}"
        f"_deep_{deep_flow:g}.png"
    )

    output_file = (
        PLOT_DIR
        / filename
    )

    figure.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"  Physical trials: "
        f"{len(columns)}"
    )

    print(
        f"  Overall RMSE: "
        f"{condition_rmse:.8f}"
    )

    print(
        f"  Saved plot: "
        f"{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("DCS EVALUATION ANALYSIS")
    print("=" * 80)

    # ========================================================
    # READ EVALUATION CSV
    # ========================================================

    if not RESULT_CSV.exists():
        raise FileNotFoundError(
            f"Evaluation CSV not found:\n"
            f"{RESULT_CSV}\n\n"
            f"Run evaluation.py first."
        )

    print(
        "Reading evaluation CSV:"
    )

    print(
        RESULT_CSV
    )

    df = pd.read_csv(
        RESULT_CSV
    )

    print(
        f"Loaded {len(df)} experiments."
    )

    # ========================================================
    # CONDITION ANALYSIS
    # ========================================================

    create_condition_summary(
        df
    )

    # ========================================================
    # ZERO-FLOW ANALYSIS
    # ========================================================

    analyze_zero_flow_groups(
        df
    )

    # ========================================================
    # LOAD MAT DATA
    # ========================================================

    print()
    print("=" * 80)
    print(
        "GENERATING REPRESENTATIVE CURVE PLOTS"
    )
    print("=" * 80)

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Reference MAT file "
            f"not found:\n"
            f"{REFERENCE_FILE}"
        )

    mat = loadmat(
        REFERENCE_FILE
    )

    dataset = mat[
        "Dataset_set_intra40"
    ]

    gtau = mat[
        "ave_gtau_intra40"
    ]

    print(
        f"Dataset shape: "
        f"{dataset.shape}"
    )

    print(
        f"g2 shape: "
        f"{gtau.shape}"
    )

    # ========================================================
    # REPRESENTATIVE CONDITIONS
    # ========================================================

    for (
        shallow_flow,
        deep_flow,
        label,
    ) in SELECTED_CONDITIONS:

        print()

        print(
            f"Condition: "
            f"Shallow={shallow_flow:g}, "
            f"Deep={deep_flow:g} "
            f"({label})"
        )

        plot_condition(
            dataset=
                dataset,

            gtau=
                gtau,

            shallow_flow=
                shallow_flow,

            deep_flow=
                deep_flow,

            label=
                label,
        )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        "Condition summary:"
    )

    print(
        CONDITION_CSV
    )

    print()

    print(
        "Plots:"
    )

    print(
        PLOT_DIR
    )


if __name__ == "__main__":
    main()