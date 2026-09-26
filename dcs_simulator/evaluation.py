from pathlib import Path
import csv

import numpy as np
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

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_CSV = (
    OUTPUT_DIR
    / "full_dataset_evaluation.csv"
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    reference_g2,
    predicted_g2,
):
    reference_g2 = np.asarray(
        reference_g2,
        dtype=float,
    )

    predicted_g2 = np.asarray(
        predicted_g2,
        dtype=float,
    )

    error = (
        predicted_g2
        - reference_g2
    )

    squared_error = error ** 2

    rmse = float(
        np.sqrt(
            np.mean(
                squared_error
            )
        )
    )

    # R² used here follows the paper:
    # square of Pearson correlation coefficient
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

    return {
        "rmse": rmse,
        "r_squared": r_squared,
        "squared_error": squared_error,
    }


# ============================================================
# OVERALL RMSE
# ============================================================

def calculate_overall_rmse(
    experiment_rmse_values,
):
    values = np.asarray(
        experiment_rmse_values,
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


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    title,
    results,
):
    print()
    print("=" * 82)
    print(title)
    print("=" * 82)

    if len(results) == 0:
        print("No experiments.")
        return

    experiment_rmse = np.array(
        [
            row["experiment_rmse"]
            for row in results
        ],
        dtype=float,
    )

    overall_rmse = (
        calculate_overall_rmse(
            experiment_rmse
        )
    )

    print(
        f"Number of experiments: "
        f"{len(results)}"
    )

    print(
        f"Mean experiment RMSE:  "
        f"{np.mean(experiment_rmse):.8f}"
    )

    print(
        f"Minimum RMSE:          "
        f"{np.min(experiment_rmse):.8f}"
    )

    print(
        f"Maximum RMSE:          "
        f"{np.max(experiment_rmse):.8f}"
    )

    print(
        f"Overall RMSE:          "
        f"{overall_rmse:.8f}"
    )

    print()

    for sds_cm in (1, 2, 3):

        rmse_key = (
            f"sds{sds_cm}_rmse"
        )

        r2_key = (
            f"sds{sds_cm}_r2"
        )

        sds_rmse = np.array(
            [
                row[rmse_key]
                for row in results
            ],
            dtype=float,
        )

        sds_r2 = np.array(
            [
                row[r2_key]
                for row in results
            ],
            dtype=float,
        )

        valid_r2 = sds_r2[
            np.isfinite(sds_r2)
        ]

        print(
            f"SDS {sds_cm} cm"
        )

        print(
            f"  Mean RMSE: "
            f"{np.mean(sds_rmse):.8f}"
        )

        print(
            f"  Overall RMSE: "
            f"{calculate_overall_rmse(sds_rmse):.8f}"
        )

        if len(valid_r2) > 0:
            print(
                f"  Mean R²: "
                f"{np.mean(valid_r2):.8f}"
            )
        else:
            print(
                "  Mean R²: N/A"
            )

        print()

    # --------------------------------------------------------
    # Best / worst experiments
    # --------------------------------------------------------

    sorted_results = sorted(
        results,
        key=lambda row:
            row["experiment_rmse"],
    )

    print(
        "Lowest-error experiments:"
    )

    for row in sorted_results[:5]:
        print(
            f"  Exp {row['experiment']:4d} "
            f"| S={row['commanded_shallow']:8.3f} "
            f"D={row['commanded_deep']:8.3f} "
            f"| RMSE="
            f"{row['experiment_rmse']:.8f}"
        )

    print()

    print(
        "Highest-error experiments:"
    )

    for row in sorted_results[-5:][::-1]:
        print(
            f"  Exp {row['experiment']:4d} "
            f"| S={row['commanded_shallow']:8.3f} "
            f"D={row['commanded_deep']:8.3f} "
            f"| RMSE="
            f"{row['experiment_rmse']:.8f}"
        )

    print("=" * 82)


# ============================================================
# SAVE CSV
# ============================================================

def save_results_csv(
    results,
):
    fieldnames = [
        "experiment",
        "mat_column",

        "commanded_shallow",
        "commanded_deep",

        "measured_shallow",
        "measured_deep",

        "is_zero_zero",

        "sds1_rmse",
        "sds1_r2",

        "sds2_rmse",
        "sds2_r2",

        "sds3_rmse",
        "sds3_r2",

        "experiment_rmse",
    ]

    with open(
        RESULT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# ============================================================
# FULL DATASET EVALUATION
# ============================================================

def evaluate_full_dataset():

    print("=" * 82)
    print(
        "FULL DATASET EVALUATION"
    )
    print("=" * 82)

    print(
        f"Loading: "
        f"{REFERENCE_FILE}"
    )

    data = loadmat(
        REFERENCE_FILE
    )

    dataset = data[
        "Dataset_set_intra40"
    ]

    gtau = data[
        "ave_gtau_intra40"
    ]

    number_of_records = (
        dataset.shape[1]
    )

    print(
        f"Dataset records: "
        f"{number_of_records}"
    )

    print(
        f"g2 shape: "
        f"{gtau.shape}"
    )

    print()

    results = []

    # --------------------------------------------------------
    # Cache model results
    #
    # Same commanded shallow/deep condition produces the same
    # model curve, so there is no need to regenerate the model
    # for every repeated physical trial.
    # --------------------------------------------------------

    model_cache = {}

    # ========================================================
    # LOOP THROUGH ALL PHYSICAL RECORDS
    # ========================================================

    for column_index in range(
        number_of_records
    ):

        experiment_number = (
            column_index + 1
        )

        commanded_shallow = float(
            dataset[
                0,
                column_index,
            ]
        )

        commanded_deep = float(
            dataset[
                1,
                column_index,
            ]
        )

        measured_shallow = float(
            dataset[
                2,
                column_index,
            ]
        )

        measured_deep = float(
            dataset[
                3,
                column_index,
            ]
        )

        # ----------------------------------------------------
        # Check (0,0)
        # ----------------------------------------------------

        is_zero_zero = bool(
            np.isclose(
                commanded_shallow,
                0.0,
            )
            and np.isclose(
                commanded_deep,
                0.0,
            )
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        condition_key = (
            commanded_shallow,
            commanded_deep,
        )

        if (
            condition_key
            not in model_cache
        ):
            model_cache[
                condition_key
            ] = run_validation(
                shallow_flow=
                    commanded_shallow,

                deep_flow=
                    commanded_deep,
            )

        model = model_cache[
            condition_key
        ]

        # ----------------------------------------------------
        # Evaluate SDS 1, 2, 3
        # ----------------------------------------------------

        sds_metrics = {}

        experiment_squared_errors = []

        for sds_cm in (1, 2, 3):

            physical_g2 = np.asarray(
                gtau[
                    sds_cm - 1,
                    :,
                    column_index,
                ],
                dtype=float,
            )

            model_g2 = np.asarray(
                model[
                    "results"
                ][
                    sds_cm
                ][
                    "generated_g2"
                ],
                dtype=float,
            )

            if (
                physical_g2.shape
                != model_g2.shape
            ):
                raise ValueError(
                    f"Shape mismatch at "
                    f"experiment "
                    f"{experiment_number}, "
                    f"SDS {sds_cm}: "
                    f"physical="
                    f"{physical_g2.shape}, "
                    f"model="
                    f"{model_g2.shape}"
                )

            metrics = (
                calculate_metrics(
                    physical_g2,
                    model_g2,
                )
            )

            sds_metrics[
                sds_cm
            ] = metrics

            experiment_squared_errors.extend(
                metrics[
                    "squared_error"
                ]
            )

        # ----------------------------------------------------
        # One RMSE per experiment
        #
        # 3 SDS x 501 points = 1503 pointwise errors
        # ----------------------------------------------------

        experiment_squared_errors = (
            np.asarray(
                experiment_squared_errors,
                dtype=float,
            )
        )

        experiment_rmse = float(
            np.sqrt(
                np.mean(
                    experiment_squared_errors
                )
            )
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {
            "experiment":
                experiment_number,

            "mat_column":
                column_index,

            "commanded_shallow":
                commanded_shallow,

            "commanded_deep":
                commanded_deep,

            "measured_shallow":
                measured_shallow,

            "measured_deep":
                measured_deep,

            "is_zero_zero":
                is_zero_zero,

            "sds1_rmse":
                sds_metrics[1]["rmse"],

            "sds1_r2":
                sds_metrics[1]["r_squared"],

            "sds2_rmse":
                sds_metrics[2]["rmse"],

            "sds2_r2":
                sds_metrics[2]["r_squared"],

            "sds3_rmse":
                sds_metrics[3]["rmse"],

            "sds3_r2":
                sds_metrics[3]["r_squared"],

            "experiment_rmse":
                experiment_rmse,
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            experiment_number == 1
            or experiment_number % 100 == 0
            or experiment_number
            == number_of_records
        ):
            print(
                f"Processed "
                f"{experiment_number}"
                f"/"
                f"{number_of_records}"
                f" records"
            )

    # ========================================================
    # SAVE CSV
    # ========================================================

    save_results_csv(
        results
    )

    # ========================================================
    # ALL RECORDS
    # ========================================================

    print_summary(
        title=(
            "SUMMARY: ALL RECORDS "
            "(INCLUDING 0,0)"
        ),
        results=results,
    )

    # ========================================================
    # EXCLUDE (0,0)
    # ========================================================

    non_zero_results = [
        row
        for row in results
        if not row[
            "is_zero_zero"
        ]
    ]

    print_summary(
        title=(
            "SUMMARY: EXCLUDING "
            "(0,0)"
        ),
        results=non_zero_results,
    )

    # ========================================================
    # (0,0) ONLY
    # ========================================================

    zero_results = [
        row
        for row in results
        if row[
            "is_zero_zero"
        ]
    ]

    print_summary(
        title=(
            "SUMMARY: (0,0) ONLY"
        ),
        results=zero_results,
    )

    # ========================================================
    # FINISH
    # ========================================================

    print()

    print(
        f"CSV saved to:"
    )

    print(
        RESULT_CSV
    )

    print()

    print(
        "Evaluation complete."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    evaluate_full_dataset()