"""
Model Comparison and Operating Region Analysis

Purpose
-------
Compare the current Proposed Model with the temporary/older Baseline Model
across all available Physical Phantom flow conditions.

This script is intended to answer questions such as:

- In which flow conditions does the Proposed Model perform better?
- In which flow conditions does the Baseline Model perform better?
- Does either model perform better at low/high shallow flow?
- Does either model perform better at low/high deep flow?
- Are there regions where both models have similar accuracy?
- Are there possible crossover regions between the two models?

IMPORTANT
---------
The current Baseline Model is an older/temporary teammate baseline.
Results from this script should therefore be treated as a DEMO of the
comparison methodology until the updated teammate model is available.

The Physical Reference currently follows the same temporary fitted reference
used by the GUI validation pipeline.

Outputs
-------
output/model_comparison/
    model_comparison_by_condition.csv
    model_comparison_by_shallow.csv
    model_comparison_by_deep.csv
    model_comparison_by_sds.csv
    crossover_candidates.csv
    heatmaps/
    trends/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

from gui import run_validation
from reference_loader import REFERENCE_FILE


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "model_comparison"
)

HEATMAP_DIR = (
    OUTPUT_DIR
    / "heatmaps"
)

TREND_DIR = (
    OUTPUT_DIR
    / "trends"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

HEATMAP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TREND_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

# Difference between model RMSE values that will be considered
# "approximately similar".
#
# This is NOT an acceptance threshold for the model.
# It is used only for finding possible crossover / transition regions.
CROSSOVER_TOLERANCE = 0.005

# Exclude (0,0) from the main operating-region interpretation because
# previous evaluation showed that this condition behaves very differently.
#
# The condition is still evaluated and remains in the full CSV.
EXCLUDE_ZERO_ZERO_FROM_SUMMARY = True


# ============================================================
# HELPERS
# ============================================================

def format_flow(value):
    """
    Format flow value for console output and plot labels.
    """

    value = float(value)

    if value.is_integer():
        return str(int(value))

    return f"{value:g}"


def rmse_difference(
    proposed_rmse,
    baseline_rmse,
):
    """
    Delta RMSE definition:

        delta = Proposed RMSE - Baseline RMSE

    delta < 0  -> Proposed is better
    delta > 0  -> Baseline is better
    delta ~= 0 -> Similar performance
    """

    return float(
        proposed_rmse
        - baseline_rmse
    )


def classify_model(
    proposed_rmse,
    baseline_rmse,
    tolerance=CROSSOVER_TOLERANCE,
):
    """
    Classify which model performs better.

    Similar:
        absolute RMSE difference <= tolerance

    Proposed:
        Proposed RMSE is lower

    Baseline:
        Baseline RMSE is lower
    """

    difference = rmse_difference(
        proposed_rmse,
        baseline_rmse,
    )

    if abs(difference) <= tolerance:
        return "Similar"

    if difference < 0:
        return "Proposed"

    return "Baseline"


def calculate_percent_difference(
    proposed_rmse,
    baseline_rmse,
):
    """
    Relative difference between the two RMSE values.

    Positive:
        Proposed has lower RMSE.

    Negative:
        Baseline has lower RMSE.
    """

    proposed_rmse = float(
        proposed_rmse
    )

    baseline_rmse = float(
        baseline_rmse
    )

    denominator = max(
        proposed_rmse,
        baseline_rmse,
    )

    if denominator == 0:
        return 0.0

    return float(
        (
            baseline_rmse
            - proposed_rmse
        )
        / denominator
        * 100.0
    )


# ============================================================
# LOAD FLOW CONDITIONS
# ============================================================

def load_all_conditions():
    """
    Load all unique commanded shallow/deep flow conditions
    from Dataset_phantom-intra40.mat.

    Dataset_set_intra40:
        row 0 = commanded shallow flow
        row 1 = commanded deep flow
    """

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            "Reference MAT file not found:\n"
            f"{REFERENCE_FILE}"
        )

    mat = loadmat(
        REFERENCE_FILE
    )

    if "Dataset_set_intra40" not in mat:
        raise KeyError(
            "Dataset_set_intra40 "
            "not found in MAT file."
        )

    dataset = np.asarray(
        mat["Dataset_set_intra40"],
        dtype=float,
    )

    if dataset.ndim != 2:
        raise ValueError(
            "Expected Dataset_set_intra40 "
            f"to be 2D. Got {dataset.shape}"
        )

    if dataset.shape[0] < 2:
        raise ValueError(
            "Dataset_set_intra40 must "
            "contain at least 2 rows."
        )

    shallow = dataset[0]
    deep = dataset[1]

    conditions = sorted(
        {
            (
                float(s),
                float(d),
            )
            for s, d
            in zip(
                shallow,
                deep,
            )
        },
        key=lambda item: (
            item[0],
            item[1],
        ),
    )

    return conditions


# ============================================================
# RUN ONE CONDITION
# ============================================================

def evaluate_condition(
    shallow_flow,
    deep_flow,
):
    """
    Run Proposed and Baseline models for one condition.

    Uses the same validation pipeline as the GUI so that:

    - Proposed model
    - Baseline model
    - Temporary Physical Fit

    are generated consistently.
    """

    output = run_validation(
        shallow_flow=shallow_flow,
        deep_flow=deep_flow,
    )

    results = output[
        "results"
    ]

    row = {
        "shallow_flow": float(
            shallow_flow
        ),
        "deep_flow": float(
            deep_flow
        ),
    }

    # --------------------------------------------------------
    # SDS-specific metrics
    # --------------------------------------------------------

    for sds_cm in (
        1,
        2,
        3,
    ):

        proposed_rmse = float(
            results[sds_cm][
                "rmse"
            ]
        )

        baseline_rmse = float(
            results[sds_cm][
                "baseline_rmse"
            ]
        )

        delta = rmse_difference(
            proposed_rmse,
            baseline_rmse,
        )

        better = classify_model(
            proposed_rmse,
            baseline_rmse,
        )

        percent_difference = (
            calculate_percent_difference(
                proposed_rmse,
                baseline_rmse,
            )
        )

        row[
            f"sds{sds_cm}_proposed_rmse"
        ] = proposed_rmse

        row[
            f"sds{sds_cm}_baseline_rmse"
        ] = baseline_rmse

        row[
            f"sds{sds_cm}_delta_rmse"
        ] = delta

        row[
            f"sds{sds_cm}_better"
        ] = better

        row[
            f"sds{sds_cm}_percent_difference"
        ] = percent_difference

    # --------------------------------------------------------
    # Combined metrics
    # --------------------------------------------------------

    proposed_combined = float(
        output[
            "proposed_combined_rmse"
        ]
    )

    baseline_combined = float(
        output[
            "baseline_combined_rmse"
        ]
    )

    combined_delta = (
        rmse_difference(
            proposed_combined,
            baseline_combined,
        )
    )

    combined_better = (
        classify_model(
            proposed_combined,
            baseline_combined,
        )
    )

    combined_percent = (
        calculate_percent_difference(
            proposed_combined,
            baseline_combined,
        )
    )

    row[
        "proposed_combined_rmse"
    ] = proposed_combined

    row[
        "baseline_combined_rmse"
    ] = baseline_combined

    row[
        "combined_delta_rmse"
    ] = combined_delta

    row[
        "combined_better"
    ] = combined_better

    row[
        "combined_percent_difference"
    ] = combined_percent

    return row


# ============================================================
# RUN ALL CONDITIONS
# ============================================================

def run_all_conditions():
    """
    Evaluate every unique commanded flow condition.
    """

    conditions = (
        load_all_conditions()
    )

    print()
    print(
        "=" * 72
    )
    print(
        "MODEL COMPARISON"
    )
    print(
        "=" * 72
    )

    print(
        "Comparison:"
    )

    print(
        "  Proposed Model"
    )

    print(
        "  vs"
    )

    print(
        "  Temporary / Older Baseline Model"
    )

    print()

    print(
        "Reference:"
    )

    print(
        "  Temporary Physical Fit "
        "used by current GUI pipeline"
    )

    print()

    print(
        f"Unique conditions: "
        f"{len(conditions)}"
    )

    print(
        "Crossover tolerance: "
        f"{CROSSOVER_TOLERANCE:.6f}"
    )

    print()

    rows = []

    total = len(
        conditions
    )

    for index, (
        shallow_flow,
        deep_flow,
    ) in enumerate(
        conditions,
        start=1,
    ):

        print(
            f"[{index:3d}/{total}] "
            f"Shallow = "
            f"{format_flow(shallow_flow):>5}, "
            f"Deep = "
            f"{format_flow(deep_flow):>5}",
            end="",
            flush=True,
        )

        try:

            row = evaluate_condition(
                shallow_flow,
                deep_flow,
            )

            rows.append(
                row
            )

            print(
                "  ->  "
                f"P={row['proposed_combined_rmse']:.6f}  "
                f"B={row['baseline_combined_rmse']:.6f}  "
                f"{row['combined_better']}"
            )

        except Exception as error:

            print(
                "  -> ERROR"
            )

            print(
                f"      {error}"
            )

    if not rows:
        raise RuntimeError(
            "No conditions were evaluated successfully."
        )

    dataframe = pd.DataFrame(
        rows
    )

    dataframe = dataframe.sort_values(
        [
            "shallow_flow",
            "deep_flow",
        ]
    ).reset_index(
        drop=True
    )

    output_file = (
        OUTPUT_DIR
        / "model_comparison_by_condition.csv"
    )

    dataframe.to_csv(
        output_file,
        index=False,
    )

    print()
    print(
        "Condition comparison saved:"
    )

    print(
        output_file
    )

    return dataframe


# ============================================================
# SUMMARY FILTER
# ============================================================

def get_summary_dataframe(
    dataframe,
):
    """
    Return dataframe used for main operating-region analysis.

    (0,0) remains in the full CSV but can be excluded from
    interpretation.
    """

    if not (
        EXCLUDE_ZERO_ZERO_FROM_SUMMARY
    ):
        return (
            dataframe.copy()
        )

    mask = ~(
        np.isclose(
            dataframe[
                "shallow_flow"
            ],
            0.0,
        )
        &
        np.isclose(
            dataframe[
                "deep_flow"
            ],
            0.0,
        )
    )

    return (
        dataframe.loc[
            mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


# ============================================================
# OVERALL SUMMARY
# ============================================================

def print_overall_summary(
    dataframe,
):
    """
    Print overall comparison statistics.
    """

    analysis = (
        get_summary_dataframe(
            dataframe
        )
    )

    print()
    print(
        "=" * 72
    )
    print(
        "OVERALL OPERATING-REGION SUMMARY"
    )
    print(
        "=" * 72
    )

    if (
        EXCLUDE_ZERO_ZERO_FROM_SUMMARY
    ):
        print(
            "Main summary excludes "
            "(0,0)."
        )

    print(
        f"Conditions analyzed: "
        f"{len(analysis)}"
    )

    print()

    counts = (
        analysis[
            "combined_better"
        ]
        .value_counts()
    )

    for label in (
        "Proposed",
        "Baseline",
        "Similar",
    ):

        count = int(
            counts.get(
                label,
                0,
            )
        )

        percent = (
            count
            / len(analysis)
            * 100.0
        )

        print(
            f"{label:10s}: "
            f"{count:3d} "
            f"({percent:6.2f}%)"
        )

    print()

    print(
        "Mean Combined RMSE"
    )

    print(
        "  Proposed = "
        f"{analysis['proposed_combined_rmse'].mean():.8f}"
    )

    print(
        "  Baseline = "
        f"{analysis['baseline_combined_rmse'].mean():.8f}"
    )

    proposed_overall = float(
        np.sqrt(
            np.mean(
                analysis[
                    "proposed_combined_rmse"
                ].to_numpy() ** 2
            )
        )
    )

    baseline_overall = float(
        np.sqrt(
            np.mean(
                analysis[
                    "baseline_combined_rmse"
                ].to_numpy() ** 2
            )
        )
    )

    print()

    print(
        "RMS across condition RMSE values"
    )

    print(
        "  Proposed = "
        f"{proposed_overall:.8f}"
    )

    print(
        "  Baseline = "
        f"{baseline_overall:.8f}"
    )


# ============================================================
# SUMMARY BY SHALLOW FLOW
# ============================================================

def summarize_by_shallow(
    dataframe,
):
    """
    Analyze performance as shallow flow changes.
    """

    analysis = (
        get_summary_dataframe(
            dataframe
        )
    )

    rows = []

    for shallow_flow, group in (
        analysis.groupby(
            "shallow_flow"
        )
    ):

        proposed = group[
            "proposed_combined_rmse"
        ].to_numpy()

        baseline = group[
            "baseline_combined_rmse"
        ].to_numpy()

        delta = (
            proposed
            - baseline
        )

        proposed_count = int(
            np.sum(
                delta
                < -CROSSOVER_TOLERANCE
            )
        )

        baseline_count = int(
            np.sum(
                delta
                > CROSSOVER_TOLERANCE
            )
        )

        similar_count = int(
            np.sum(
                np.abs(delta)
                <= CROSSOVER_TOLERANCE
            )
        )

        rows.append(
            {
                "shallow_flow":
                    float(
                        shallow_flow
                    ),

                "condition_count":
                    len(group),

                "mean_proposed_rmse":
                    float(
                        np.mean(
                            proposed
                        )
                    ),

                "mean_baseline_rmse":
                    float(
                        np.mean(
                            baseline
                        )
                    ),

                "mean_delta_rmse":
                    float(
                        np.mean(
                            delta
                        )
                    ),

                "proposed_better_count":
                    proposed_count,

                "baseline_better_count":
                    baseline_count,

                "similar_count":
                    similar_count,

                "proposed_better_percent":
                    proposed_count
                    / len(group)
                    * 100.0,

                "baseline_better_percent":
                    baseline_count
                    / len(group)
                    * 100.0,

                "similar_percent":
                    similar_count
                    / len(group)
                    * 100.0,
            }
        )

    summary = pd.DataFrame(
        rows
    )

    summary = summary.sort_values(
        "shallow_flow"
    ).reset_index(
        drop=True
    )

    output_file = (
        OUTPUT_DIR
        / "model_comparison_by_shallow.csv"
    )

    summary.to_csv(
        output_file,
        index=False,
    )

    return summary


# ============================================================
# SUMMARY BY DEEP FLOW
# ============================================================

def summarize_by_deep(
    dataframe,
):
    """
    Analyze performance as deep flow changes.
    """

    analysis = (
        get_summary_dataframe(
            dataframe
        )
    )

    rows = []

    for deep_flow, group in (
        analysis.groupby(
            "deep_flow"
        )
    ):

        proposed = group[
            "proposed_combined_rmse"
        ].to_numpy()

        baseline = group[
            "baseline_combined_rmse"
        ].to_numpy()

        delta = (
            proposed
            - baseline
        )

        proposed_count = int(
            np.sum(
                delta
                < -CROSSOVER_TOLERANCE
            )
        )

        baseline_count = int(
            np.sum(
                delta
                > CROSSOVER_TOLERANCE
            )
        )

        similar_count = int(
            np.sum(
                np.abs(delta)
                <= CROSSOVER_TOLERANCE
            )
        )

        rows.append(
            {
                "deep_flow":
                    float(
                        deep_flow
                    ),

                "condition_count":
                    len(group),

                "mean_proposed_rmse":
                    float(
                        np.mean(
                            proposed
                        )
                    ),

                "mean_baseline_rmse":
                    float(
                        np.mean(
                            baseline
                        )
                    ),

                "mean_delta_rmse":
                    float(
                        np.mean(
                            delta
                        )
                    ),

                "proposed_better_count":
                    proposed_count,

                "baseline_better_count":
                    baseline_count,

                "similar_count":
                    similar_count,

                "proposed_better_percent":
                    proposed_count
                    / len(group)
                    * 100.0,

                "baseline_better_percent":
                    baseline_count
                    / len(group)
                    * 100.0,

                "similar_percent":
                    similar_count
                    / len(group)
                    * 100.0,
            }
        )

    summary = pd.DataFrame(
        rows
    )

    summary = summary.sort_values(
        "deep_flow"
    ).reset_index(
        drop=True
    )

    output_file = (
        OUTPUT_DIR
        / "model_comparison_by_deep.csv"
    )

    summary.to_csv(
        output_file,
        index=False,
    )

    return summary


# ============================================================
# SUMMARY BY SDS
# ============================================================

def summarize_by_sds(
    dataframe,
):
    """
    Compare Proposed and Baseline performance for each SDS.
    """

    analysis = (
        get_summary_dataframe(
            dataframe
        )
    )

    rows = []

    for sds_cm in (
        1,
        2,
        3,
    ):

        proposed = analysis[
            f"sds{sds_cm}_proposed_rmse"
        ].to_numpy()

        baseline = analysis[
            f"sds{sds_cm}_baseline_rmse"
        ].to_numpy()

        delta = (
            proposed
            - baseline
        )

        proposed_count = int(
            np.sum(
                delta
                < -CROSSOVER_TOLERANCE
            )
        )

        baseline_count = int(
            np.sum(
                delta
                > CROSSOVER_TOLERANCE
            )
        )

        similar_count = int(
            np.sum(
                np.abs(delta)
                <= CROSSOVER_TOLERANCE
            )
        )

        rows.append(
            {
                "sds_cm":
                    sds_cm,

                "condition_count":
                    len(analysis),

                "mean_proposed_rmse":
                    float(
                        np.mean(
                            proposed
                        )
                    ),

                "mean_baseline_rmse":
                    float(
                        np.mean(
                            baseline
                        )
                    ),

                "mean_delta_rmse":
                    float(
                        np.mean(
                            delta
                        )
                    ),

                "proposed_better_count":
                    proposed_count,

                "baseline_better_count":
                    baseline_count,

                "similar_count":
                    similar_count,

                "proposed_better_percent":
                    proposed_count
                    / len(analysis)
                    * 100.0,

                "baseline_better_percent":
                    baseline_count
                    / len(analysis)
                    * 100.0,

                "similar_percent":
                    similar_count
                    / len(analysis)
                    * 100.0,
            }
        )

    summary = pd.DataFrame(
        rows
    )

    output_file = (
        OUTPUT_DIR
        / "model_comparison_by_sds.csv"
    )

    summary.to_csv(
        output_file,
        index=False,
    )

    return summary


# ============================================================
# CROSSOVER CANDIDATES
# ============================================================

def find_crossover_candidates(
    dataframe,
):
    """
    Find conditions where Proposed and Baseline have similar RMSE.

    These are candidate transition/crossover regions.

    This does NOT mean that the two mathematical models literally
    intersect at exactly one point. It means their overall accuracy
    for the condition is similar.
    """

    analysis = (
        get_summary_dataframe(
            dataframe
        )
    )

    crossover = analysis[
        np.abs(
            analysis[
                "combined_delta_rmse"
            ]
        )
        <= CROSSOVER_TOLERANCE
    ].copy()

    crossover[
        "absolute_delta_rmse"
    ] = np.abs(
        crossover[
            "combined_delta_rmse"
        ]
    )

    crossover = (
        crossover
        .sort_values(
            "absolute_delta_rmse"
        )
        .reset_index(
            drop=True
        )
    )

    output_file = (
        OUTPUT_DIR
        / "crossover_candidates.csv"
    )

    crossover.to_csv(
        output_file,
        index=False,
    )

    return crossover


# ============================================================
# HEATMAP
# ============================================================

def create_heatmap(
    dataframe,
    value_column,
    title,
    filename,
):
    """
    Create a heatmap where:

        negative delta -> Proposed lower RMSE
        positive delta -> Baseline lower RMSE
        near zero      -> similar accuracy

    Rows = Shallow Flow
    Columns = Deep Flow
    """

    analysis = (
        get_summary_dataframe(
            dataframe
        )
    )

    pivot = analysis.pivot(
        index="shallow_flow",
        columns="deep_flow",
        values=value_column,
    )

    pivot = pivot.sort_index(
        axis=0
    )

    pivot = pivot.sort_index(
        axis=1
    )

    values = pivot.to_numpy(
        dtype=float
    )

    masked_values = (
        np.ma.masked_invalid(
            values
        )
    )

    figure_width = max(
        10,
        len(
            pivot.columns
        )
        * 0.45,
    )

    figure_height = max(
        6,
        len(
            pivot.index
        )
        * 0.40,
    )

    fig, ax = plt.subplots(
        figsize=(
            figure_width,
            figure_height,
        )
    )

    max_abs = np.nanmax(
        np.abs(
            values
        )
    )

    if (
        not np.isfinite(
            max_abs
        )
        or max_abs == 0
    ):
        max_abs = 1.0

    image = ax.imshow(
        masked_values,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )

    ax.set_xticks(
        np.arange(
            len(
                pivot.columns
            )
        )
    )

    ax.set_xticklabels(
        [
            format_flow(
                value
            )
            for value
            in pivot.columns
        ],
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        np.arange(
            len(
                pivot.index
            )
        )
    )

    ax.set_yticklabels(
        [
            format_flow(
                value
            )
            for value
            in pivot.index
        ]
    )

    ax.set_xlabel(
        "Deep Flow"
    )

    ax.set_ylabel(
        "Shallow Flow"
    )

    ax.set_title(
        title
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "ΔRMSE = Proposed - Baseline\n"
        "Negative: Proposed better | "
        "Positive: Baseline better"
    )

    # Mark near-crossover cells
    for row_index in range(
        values.shape[0]
    ):

        for column_index in range(
            values.shape[1]
        ):

            value = values[
                row_index,
                column_index,
            ]

            if not np.isfinite(
                value
            ):
                continue

            if (
                abs(value)
                <= CROSSOVER_TOLERANCE
            ):

                ax.text(
                    column_index,
                    row_index,
                    "×",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

    fig.tight_layout()

    output_file = (
        HEATMAP_DIR
        / filename
    )

    fig.savefig(
        output_file,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_file


def create_all_heatmaps(
    dataframe,
):
    """
    Create Combined + SDS-specific heatmaps.
    """

    files = []

    files.append(
        create_heatmap(
            dataframe=dataframe,
            value_column=(
                "combined_delta_rmse"
            ),
            title=(
                "Combined Model Performance Map\n"
                "ΔRMSE = Proposed - Baseline"
            ),
            filename=(
                "combined_delta_rmse_heatmap.png"
            ),
        )
    )

    for sds_cm in (
        1,
        2,
        3,
    ):

        files.append(
            create_heatmap(
                dataframe=dataframe,
                value_column=(
                    f"sds{sds_cm}_delta_rmse"
                ),
                title=(
                    f"SDS {sds_cm} cm "
                    "Model Performance Map\n"
                    "ΔRMSE = Proposed - Baseline"
                ),
                filename=(
                    f"sds{sds_cm}_delta_rmse_heatmap.png"
                ),
            )
        )

    return files


# ============================================================
# TREND PLOTS
# ============================================================

def create_shallow_fixed_trends(
    dataframe,
):
    """
    For each Shallow Flow value:

        X = Deep Flow
        Y = Combined RMSE

    This makes it possible to see whether Proposed and Baseline
    accuracy trends approach or cross each other as Deep Flow changes.
    """

    analysis = (
        get_summary_dataframe(
            dataframe
        )
    )

    output_files = []

    for shallow_flow, group in (
        analysis.groupby(
            "shallow_flow"
        )
    ):

        group = group.sort_values(
            "deep_flow"
        )

        if len(group) < 2:
            continue

        fig, ax = plt.subplots(
            figsize=(
                8,
                5,
            )
        )

        ax.plot(
            group[
                "deep_flow"
            ],
            group[
                "proposed_combined_rmse"
            ],
            marker="o",
            linewidth=2,
            label="Proposed",
        )

        ax.plot(
            group[
                "deep_flow"
            ],
            group[
                "baseline_combined_rmse"
            ],
            marker="o",
            linewidth=2,
            label="Baseline",
        )

        similar = group[
            np.abs(
                group[
                    "combined_delta_rmse"
                ]
            )
            <= CROSSOVER_TOLERANCE
        ]

        if not similar.empty:

            ax.scatter(
                similar[
                    "deep_flow"
                ],
                (
                    similar[
                        "proposed_combined_rmse"
                    ]
                    +
                    similar[
                        "baseline_combined_rmse"
                    ]
                )
                / 2.0,
                marker="x",
                s=80,
                label="Near crossover",
            )

        ax.set_xlabel(
            "Deep Flow"
        )

        ax.set_ylabel(
            "Combined RMSE"
        )

        ax.set_title(
            "Model Accuracy Trend\n"
            f"Fixed Shallow Flow = "
            f"{format_flow(shallow_flow)}"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        fig.tight_layout()

        filename = (
            "fixed_shallow_"
            f"{format_flow(shallow_flow)}"
            ".png"
        )

        output_file = (
            TREND_DIR
            / filename
        )

        fig.savefig(
            output_file,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

        output_files.append(
            output_file
        )

    return output_files


def create_deep_fixed_trends(
    dataframe,
):
    """
    For each Deep Flow value:

        X = Shallow Flow
        Y = Combined RMSE

    This makes it possible to see whether Proposed and Baseline
    accuracy trends approach or cross each other as Shallow Flow changes.
    """

    analysis = (
        get_summary_dataframe(
            dataframe
        )
    )

    output_files = []

    for deep_flow, group in (
        analysis.groupby(
            "deep_flow"
        )
    ):

        group = group.sort_values(
            "shallow_flow"
        )

        if len(group) < 2:
            continue

        fig, ax = plt.subplots(
            figsize=(
                8,
                5,
            )
        )

        ax.plot(
            group[
                "shallow_flow"
            ],
            group[
                "proposed_combined_rmse"
            ],
            marker="o",
            linewidth=2,
            label="Proposed",
        )

        ax.plot(
            group[
                "shallow_flow"
            ],
            group[
                "baseline_combined_rmse"
            ],
            marker="o",
            linewidth=2,
            label="Baseline",
        )

        similar = group[
            np.abs(
                group[
                    "combined_delta_rmse"
                ]
            )
            <= CROSSOVER_TOLERANCE
        ]

        if not similar.empty:

            ax.scatter(
                similar[
                    "shallow_flow"
                ],
                (
                    similar[
                        "proposed_combined_rmse"
                    ]
                    +
                    similar[
                        "baseline_combined_rmse"
                    ]
                )
                / 2.0,
                marker="x",
                s=80,
                label="Near crossover",
            )

        ax.set_xlabel(
            "Shallow Flow"
        )

        ax.set_ylabel(
            "Combined RMSE"
        )

        ax.set_title(
            "Model Accuracy Trend\n"
            f"Fixed Deep Flow = "
            f"{format_flow(deep_flow)}"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        fig.tight_layout()

        filename = (
            "fixed_deep_"
            f"{format_flow(deep_flow)}"
            ".png"
        )

        output_file = (
            TREND_DIR
            / filename
        )

        fig.savefig(
            output_file,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

        output_files.append(
            output_file
        )

    return output_files


# ============================================================
# PRINT MOST DISTINCT CONDITIONS
# ============================================================

def print_extreme_conditions(
    dataframe,
    top_n=10,
):
    """
    Print conditions where each model has the largest advantage.
    """

    analysis = (
        get_summary_dataframe(
            dataframe
        )
    )

    print()
    print(
        "=" * 72
    )
    print(
        "CONDITIONS WHERE PROPOSED "
        "HAS LARGEST ADVANTAGE"
    )
    print(
        "=" * 72
    )

    proposed_best = (
        analysis
        .sort_values(
            "combined_delta_rmse",
            ascending=True,
        )
        .head(
            top_n
        )
    )

    for _, row in (
        proposed_best.iterrows()
    ):

        print(
            "Shallow "
            f"{format_flow(row['shallow_flow']):>5}  "
            "Deep "
            f"{format_flow(row['deep_flow']):>5}  "
            f"P={row['proposed_combined_rmse']:.6f}  "
            f"B={row['baseline_combined_rmse']:.6f}  "
            f"Δ={row['combined_delta_rmse']:+.6f}"
        )

    print()
    print(
        "=" * 72
    )
    print(
        "CONDITIONS WHERE BASELINE "
        "HAS LARGEST ADVANTAGE"
    )
    print(
        "=" * 72
    )

    baseline_best = (
        analysis
        .sort_values(
            "combined_delta_rmse",
            ascending=False,
        )
        .head(
            top_n
        )
    )

    for _, row in (
        baseline_best.iterrows()
    ):

        print(
            "Shallow "
            f"{format_flow(row['shallow_flow']):>5}  "
            "Deep "
            f"{format_flow(row['deep_flow']):>5}  "
            f"P={row['proposed_combined_rmse']:.6f}  "
            f"B={row['baseline_combined_rmse']:.6f}  "
            f"Δ={row['combined_delta_rmse']:+.6f}"
        )


# ============================================================
# PRINT CROSSOVER CANDIDATES
# ============================================================

def print_crossover_candidates(
    crossover,
    top_n=20,
):
    """
    Print conditions with the smallest difference between models.
    """

    print()
    print(
        "=" * 72
    )
    print(
        "POSSIBLE CROSSOVER / "
        "SIMILAR-PERFORMANCE CONDITIONS"
    )
    print(
        "=" * 72
    )

    print(
        "Definition:"
    )

    print(
        "|Proposed RMSE - Baseline RMSE| "
        f"<= {CROSSOVER_TOLERANCE:.6f}"
    )

    print()

    if crossover.empty:

        print(
            "No conditions found "
            "within current tolerance."
        )

        return

    print(
        f"Candidates found: "
        f"{len(crossover)}"
    )

    print()

    for _, row in (
        crossover
        .head(
            top_n
        )
        .iterrows()
    ):

        print(
            "Shallow "
            f"{format_flow(row['shallow_flow']):>5}  "
            "Deep "
            f"{format_flow(row['deep_flow']):>5}  "
            f"P={row['proposed_combined_rmse']:.6f}  "
            f"B={row['baseline_combined_rmse']:.6f}  "
            f"Δ={row['combined_delta_rmse']:+.6f}"
        )


# ============================================================
# PRINT FLOW TREND SUMMARIES
# ============================================================

def print_flow_summary(
    shallow_summary,
    deep_summary,
):
    """
    Print compact shallow/deep flow summaries.
    """

    print()
    print(
        "=" * 72
    )
    print(
        "SUMMARY BY SHALLOW FLOW"
    )
    print(
        "=" * 72
    )

    for _, row in (
        shallow_summary.iterrows()
    ):

        print(
            "Shallow "
            f"{format_flow(row['shallow_flow']):>5}  "
            f"N={int(row['condition_count']):3d}  "
            f"P={row['mean_proposed_rmse']:.5f}  "
            f"B={row['mean_baseline_rmse']:.5f}  "
            f"Δ={row['mean_delta_rmse']:+.5f}  "
            f"Better count "
            f"P/B/≈ = "
            f"{int(row['proposed_better_count'])}/"
            f"{int(row['baseline_better_count'])}/"
            f"{int(row['similar_count'])}"
        )

    print()
    print(
        "=" * 72
    )
    print(
        "SUMMARY BY DEEP FLOW"
    )
    print(
        "=" * 72
    )

    for _, row in (
        deep_summary.iterrows()
    ):

        print(
            "Deep "
            f"{format_flow(row['deep_flow']):>5}  "
            f"N={int(row['condition_count']):3d}  "
            f"P={row['mean_proposed_rmse']:.5f}  "
            f"B={row['mean_baseline_rmse']:.5f}  "
            f"Δ={row['mean_delta_rmse']:+.5f}  "
            f"Better count "
            f"P/B/≈ = "
            f"{int(row['proposed_better_count'])}/"
            f"{int(row['baseline_better_count'])}/"
            f"{int(row['similar_count'])}"
        )


# ============================================================
# PRINT SDS SUMMARY
# ============================================================

def print_sds_summary(
    sds_summary,
):
    """
    Print model comparison by SDS.
    """

    print()
    print(
        "=" * 72
    )
    print(
        "SUMMARY BY SDS"
    )
    print(
        "=" * 72
    )

    for _, row in (
        sds_summary.iterrows()
    ):

        print(
            f"SDS {int(row['sds_cm'])} cm  "
            f"P mean RMSE="
            f"{row['mean_proposed_rmse']:.6f}  "
            f"B mean RMSE="
            f"{row['mean_baseline_rmse']:.6f}  "
            f"Δ="
            f"{row['mean_delta_rmse']:+.6f}  "
            f"Better count "
            f"P/B/≈ = "
            f"{int(row['proposed_better_count'])}/"
            f"{int(row['baseline_better_count'])}/"
            f"{int(row['similar_count'])}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Run every flow condition
    # --------------------------------------------------------

    comparison = (
        run_all_conditions()
    )

    # --------------------------------------------------------
    # 2. Overall summary
    # --------------------------------------------------------

    print_overall_summary(
        comparison
    )

    # --------------------------------------------------------
    # 3. Flow-region summaries
    # --------------------------------------------------------

    shallow_summary = (
        summarize_by_shallow(
            comparison
        )
    )

    deep_summary = (
        summarize_by_deep(
            comparison
        )
    )

    sds_summary = (
        summarize_by_sds(
            comparison
        )
    )

    print_flow_summary(
        shallow_summary,
        deep_summary,
    )

    print_sds_summary(
        sds_summary
    )

    # --------------------------------------------------------
    # 4. Crossover candidates
    # --------------------------------------------------------

    crossover = (
        find_crossover_candidates(
            comparison
        )
    )

    print_crossover_candidates(
        crossover
    )

    # --------------------------------------------------------
    # 5. Largest model advantages
    # --------------------------------------------------------

    print_extreme_conditions(
        comparison
    )

    # --------------------------------------------------------
    # 6. Heatmaps
    # --------------------------------------------------------

    print()
    print(
        "=" * 72
    )
    print(
        "CREATING HEATMAPS"
    )
    print(
        "=" * 72
    )

    heatmap_files = (
        create_all_heatmaps(
            comparison
        )
    )

    for file in heatmap_files:
        print(
            file
        )

    # --------------------------------------------------------
    # 7. Trend plots
    # --------------------------------------------------------

    print()
    print(
        "=" * 72
    )
    print(
        "CREATING TREND / "
        "CROSSOVER PLOTS"
    )
    print(
        "=" * 72
    )

    shallow_trend_files = (
        create_shallow_fixed_trends(
            comparison
        )
    )

    deep_trend_files = (
        create_deep_fixed_trends(
            comparison
        )
    )

    print(
        "Fixed-shallow plots: "
        f"{len(shallow_trend_files)}"
    )

    print(
        "Fixed-deep plots: "
        f"{len(deep_trend_files)}"
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print(
        "=" * 72
    )
    print(
        "MODEL COMPARISON COMPLETE"
    )
    print(
        "=" * 72
    )

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "Main files:"
    )

    print(
        "  model_comparison_by_condition.csv"
    )

    print(
        "  model_comparison_by_shallow.csv"
    )

    print(
        "  model_comparison_by_deep.csv"
    )

    print(
        "  model_comparison_by_sds.csv"
    )

    print(
        "  crossover_candidates.csv"
    )

    print(
        "  heatmaps/"
    )

    print(
        "  trends/"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "The Baseline Model used in this analysis "
        "is temporary/older."
    )

    print(
        "Rerun this analysis after the updated "
        "teammate model becomes available."
    )


if __name__ == "__main__":
    main()