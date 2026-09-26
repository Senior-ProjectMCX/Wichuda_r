from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.io import loadmat


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

CONDITION_OUTPUT = (
    OUTPUT_DIR
    / "physical_variability_by_condition.csv"
)

PAIR_OUTPUT = (
    OUTPUT_DIR
    / "physical_pairwise_rmse.csv"
)


# ============================================================
# RMSE
# ============================================================

def calculate_rmse(
    curve_a,
    curve_b,
):
    curve_a = np.asarray(
        curve_a,
        dtype=float,
    )

    curve_b = np.asarray(
        curve_b,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    curve_a
                    - curve_b
                ) ** 2
            )
        )
    )


def calculate_experiment_rmse(
    trial_a,
    trial_b,
):
    """
    Compare two physical trials using all:
        3 SDS x 501 points = 1503 points
    """

    squared_errors = []

    for sds_index in (
        0,
        1,
        2,
    ):

        curve_a = np.asarray(
            trial_a[
                sds_index
            ],
            dtype=float,
        )

        curve_b = np.asarray(
            trial_b[
                sds_index
            ],
            dtype=float,
        )

        squared_errors.extend(
            (
                curve_a
                - curve_b
            ) ** 2
        )

    squared_errors = np.asarray(
        squared_errors,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                squared_errors
            )
        )
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 80)
    print("PHYSICAL-TO-PHYSICAL VARIABILITY ANALYSIS")
    print("=" * 80)

    print(
        f"Loading:\n{REFERENCE_FILE}"
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
        f"\nDataset shape: {dataset.shape}"
    )

    print(
        f"g2 shape: {gtau.shape}"
    )

    # --------------------------------------------------------
    # Group record indices by commanded condition
    # --------------------------------------------------------

    conditions = {}

    number_of_records = (
        dataset.shape[1]
    )

    for column in range(
        number_of_records
    ):

        shallow = float(
            dataset[
                0,
                column,
            ]
        )

        deep = float(
            dataset[
                1,
                column,
            ]
        )

        key = (
            shallow,
            deep,
        )

        if key not in conditions:
            conditions[
                key
            ] = []

        conditions[
            key
        ].append(
            column
        )

    print(
        f"Unique conditions: "
        f"{len(conditions)}"
    )

    # --------------------------------------------------------
    # Pairwise comparison
    # --------------------------------------------------------

    pair_rows = []
    condition_rows = []

    for condition_number, (
        (
            shallow,
            deep,
        ),
        columns,
    ) in enumerate(
        conditions.items(),
        start=1,
    ):

        pair_experiment_rmse = []

        pair_sds_rmse = {
            1: [],
            2: [],
            3: [],
        }

        # Every unique pair of physical trials
        for (
            trial_index_a,
            trial_index_b,
        ) in combinations(
            columns,
            2,
        ):

            # Shape:
            # first 3 channels x 501 points
            trial_a = np.asarray(
                gtau[
                    0:3,
                    :,
                    trial_index_a,
                ],
                dtype=float,
            )

            trial_b = np.asarray(
                gtau[
                    0:3,
                    :,
                    trial_index_b,
                ],
                dtype=float,
            )

            # --------------------------------------------
            # Per-SDS physical-to-physical RMSE
            # --------------------------------------------

            sds_values = {}

            for sds_cm in (
                1,
                2,
                3,
            ):

                rmse = calculate_rmse(
                    trial_a[
                        sds_cm - 1
                    ],
                    trial_b[
                        sds_cm - 1
                    ],
                )

                pair_sds_rmse[
                    sds_cm
                ].append(
                    rmse
                )

                sds_values[
                    sds_cm
                ] = rmse

            # --------------------------------------------
            # 1503-point physical-to-physical RMSE
            # --------------------------------------------

            experiment_rmse = (
                calculate_experiment_rmse(
                    trial_a,
                    trial_b,
                )
            )

            pair_experiment_rmse.append(
                experiment_rmse
            )

            pair_rows.append(
                {
                    "commanded_shallow":
                        shallow,

                    "commanded_deep":
                        deep,

                    "trial_a_column":
                        trial_index_a,

                    "trial_b_column":
                        trial_index_b,

                    "sds1_rmse":
                        sds_values[1],

                    "sds2_rmse":
                        sds_values[2],

                    "sds3_rmse":
                        sds_values[3],

                    "physical_pair_rmse":
                        experiment_rmse,
                }
            )

        # --------------------------------------------
        # Condition summary
        # --------------------------------------------

        pair_experiment_rmse = np.asarray(
            pair_experiment_rmse,
            dtype=float,
        )

        if len(
            pair_experiment_rmse
        ) == 0:
            continue

        condition_rows.append(
            {
                "commanded_shallow":
                    shallow,

                "commanded_deep":
                    deep,

                "number_of_trials":
                    len(columns),

                "number_of_pairs":
                    len(
                        pair_experiment_rmse
                    ),

                "mean_pairwise_rmse":
                    float(
                        np.mean(
                            pair_experiment_rmse
                        )
                    ),

                "median_pairwise_rmse":
                    float(
                        np.median(
                            pair_experiment_rmse
                        )
                    ),

                "overall_pairwise_rmse":
                    float(
                        np.sqrt(
                            np.mean(
                                pair_experiment_rmse
                                ** 2
                            )
                        )
                    ),

                "min_pairwise_rmse":
                    float(
                        np.min(
                            pair_experiment_rmse
                        )
                    ),

                "max_pairwise_rmse":
                    float(
                        np.max(
                            pair_experiment_rmse
                        )
                    ),

                "mean_sds1_rmse":
                    float(
                        np.mean(
                            pair_sds_rmse[1]
                        )
                    ),

                "mean_sds2_rmse":
                    float(
                        np.mean(
                            pair_sds_rmse[2]
                        )
                    ),

                "mean_sds3_rmse":
                    float(
                        np.mean(
                            pair_sds_rmse[3]
                        )
                    ),
            }
        )

        if (
            condition_number == 1
            or condition_number % 20 == 0
            or condition_number
            == len(conditions)
        ):
            print(
                f"Processed "
                f"{condition_number}"
                f"/"
                f"{len(conditions)} "
                f"conditions"
            )

    # ========================================================
    # SAVE
    # ========================================================

    pair_df = pd.DataFrame(
        pair_rows
    )

    condition_df = pd.DataFrame(
        condition_rows
    )

    pair_df.to_csv(
        PAIR_OUTPUT,
        index=False,
    )

    condition_df.to_csv(
        CONDITION_OUTPUT,
        index=False,
    )

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    all_pair_rmse = pair_df[
        "physical_pair_rmse"
    ].to_numpy(
        dtype=float
    )

    print()
    print("=" * 80)
    print("GLOBAL PHYSICAL VARIABILITY")
    print("=" * 80)

    print(
        f"Physical trial pairs: "
        f"{len(all_pair_rmse)}"
    )

    print(
        f"Mean pairwise RMSE:   "
        f"{np.mean(all_pair_rmse):.8f}"
    )

    print(
        f"Median pairwise RMSE: "
        f"{np.median(all_pair_rmse):.8f}"
    )

    print(
        f"Overall pairwise RMSE:"
        f" {np.sqrt(np.mean(all_pair_rmse ** 2)):.8f}"
    )

    print(
        f"Minimum pairwise RMSE:"
        f" {np.min(all_pair_rmse):.8f}"
    )

    print(
        f"Maximum pairwise RMSE:"
        f" {np.max(all_pair_rmse):.8f}"
    )

    print()

    for percentile in (
        50,
        75,
        90,
        95,
        99,
    ):

        value = np.percentile(
            all_pair_rmse,
            percentile,
        )

        print(
            f"{percentile:2d}th percentile: "
            f"{value:.8f}"
        )

    # ========================================================
    # EXCLUDING (0,0)
    # ========================================================

    non_zero_zero = pair_df[
        ~(
            np.isclose(
                pair_df[
                    "commanded_shallow"
                ],
                0.0,
            )
            &
            np.isclose(
                pair_df[
                    "commanded_deep"
                ],
                0.0,
            )
        )
    ]

    non_zero_rmse = non_zero_zero[
        "physical_pair_rmse"
    ].to_numpy(
        dtype=float
    )

    print()
    print("=" * 80)
    print("PHYSICAL VARIABILITY EXCLUDING (0,0)")
    print("=" * 80)

    print(
        f"Physical trial pairs: "
        f"{len(non_zero_rmse)}"
    )

    print(
        f"Mean pairwise RMSE:   "
        f"{np.mean(non_zero_rmse):.8f}"
    )

    print(
        f"Median pairwise RMSE: "
        f"{np.median(non_zero_rmse):.8f}"
    )

    print(
        f"Overall pairwise RMSE:"
        f" {np.sqrt(np.mean(non_zero_rmse ** 2)):.8f}"
    )

    print()

    for percentile in (
        50,
        75,
        90,
        95,
        99,
    ):

        value = np.percentile(
            non_zero_rmse,
            percentile,
        )

        print(
            f"{percentile:2d}th percentile: "
            f"{value:.8f}"
        )

    # ========================================================
    # LOWEST / HIGHEST VARIABILITY CONDITIONS
    # ========================================================

    normal_conditions = condition_df[
        ~(
            np.isclose(
                condition_df[
                    "commanded_shallow"
                ],
                0.0,
            )
            &
            np.isclose(
                condition_df[
                    "commanded_deep"
                ],
                0.0,
            )
        )
    ]

    print()
    print("=" * 80)
    print(
        "LOWEST PHYSICAL VARIABILITY CONDITIONS "
        "(EXCLUDING 0,0)"
    )
    print("=" * 80)

    print(
        normal_conditions[
            [
                "commanded_shallow",
                "commanded_deep",
                "number_of_trials",
                "overall_pairwise_rmse",
            ]
        ].sort_values(
            "overall_pairwise_rmse"
        ).head(
            10
        ).to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print(
        "HIGHEST PHYSICAL VARIABILITY CONDITIONS "
        "(EXCLUDING 0,0)"
    )
    print("=" * 80)

    print(
        normal_conditions[
            [
                "commanded_shallow",
                "commanded_deep",
                "number_of_trials",
                "overall_pairwise_rmse",
            ]
        ].sort_values(
            "overall_pairwise_rmse",
            ascending=False,
        ).head(
            10
        ).to_string(
            index=False
        )
    )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("=" * 80)
    print("PHYSICAL VARIABILITY ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        f"Pairwise results:\n"
        f"{PAIR_OUTPUT}"
    )

    print()

    print(
        f"Condition summary:\n"
        f"{CONDITION_OUTPUT}"
    )


if __name__ == "__main__":
    main()