"""
DCS Simulator GUI

Tab 1: Reference Validation
    - Select only flow conditions available in the physical reference dataset
    - Generate model g2
    - Compare against physical reference
    - Display MAE, RMSE and maximum absolute error

Tab 2: Human Dataset Generator
    - Generate subject-specific synthetic DCS g2 datasets
    - Inputs: sex, measured thigh SAT, optional measured skin thickness
    - Flow grid is generated internally

NOTE:
The current flow -> alphaDb mapping is temporary and will be
replaced after final calibration.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import threading


import numpy as np
from scipy.io import loadmat, savemat

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from reference_loader import (
    load_reference_condition,
    REFERENCE_FILE,
)

from dcs_processor import (
    calculate_g2,
    extract_photon_data,
    K0,
)

from parameters import MU_A_MM

from baseline_model import (
    calculate_baseline_g2,
    load_baseline_pathlengths,
)

from human_model import HumanInput
from dataset_generator import generate_human_dataset




# ============================================================
# MODERN CANVAS WIDGETS
# ============================================================

def _rounded_rect(canvas, x1, y1, x2, y2, radius, **kwargs):
    radius = max(0, min(radius, (x2-x1)/2, (y2-y1)/2))
    points = [
        x1+radius, y1, x2-radius, y1, x2, y1, x2, y1+radius,
        x2, y2-radius, x2, y2, x2-radius, y2, x1+radius, y2,
        x1, y2, x1, y2-radius, x1, y1+radius, x1, y1
    ]
    return canvas.create_polygon(points, smooth=True, splinesteps=24, **kwargs)


class RoundedCard(tk.Canvas):
    def __init__(self, parent, bg_color='#FFFFFF', border_color='#E6E8F0', radius=18, **kwargs):
        # Do not lock cards to a fixed height.  A fixed canvas height clips
        # embedded controls (Entry/Combobox/Button) because Canvas does not
        # automatically grow to the requested size of its window child.
        kwargs.pop("height", None)
        super().__init__(parent, highlightthickness=0, bd=0, bg=parent.cget('bg'), **kwargs)
        self.bg_color = bg_color
        self.border_color = border_color
        self.radius = radius
        self.inner = tk.Frame(self, bg=bg_color)
        self._window = self.create_window(0, 0, anchor='nw', window=self.inner)
        self.bind('<Configure>', self._redraw)
        self.inner.bind('<Configure>', self._sync_requested_size)
        self.after_idle(self._sync_requested_size)

    def _sync_requested_size(self, event=None):
        # Let packed cards grow to fit all of their content.  Cards placed in a
        # weighted grid can still expand beyond this requested size.
        self.update_idletasks()
        requested_h = max(2, self.inner.winfo_reqheight() + 2)
        requested_w = max(2, self.inner.winfo_reqwidth() + 2)
        current = self.configure()
        try:
            if int(float(self.cget('height'))) != requested_h:
                super().configure(height=requested_h)
            # Only supply a useful minimum width; geometry managers remain free
            # to stretch the card horizontally.
            if int(float(self.cget('width'))) < requested_w:
                super().configure(width=requested_w)
        except (tk.TclError, ValueError):
            pass
        self.after_idle(self._redraw)

    def _redraw(self, event=None):
        w=max(2,self.winfo_width()); h=max(2,self.winfo_height())
        self.delete('card_bg')
        _rounded_rect(self, 1, 1, w-1, h-1, self.radius,
                      fill=self.bg_color, outline=self.border_color, width=1, tags='card_bg')
        self.tag_lower('card_bg')
        pad=1
        self.coords(self._window, pad, pad)
        self.itemconfigure(self._window, width=max(1,w-2*pad), height=max(1,h-2*pad))


class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command=None, width=180, height=42, radius=14,
                 bg='#6D5DFB', hover='#5949E8', fg='#FFFFFF', disabled_bg='#E8EAF2',
                 disabled_fg='#98A2B3', font=('Segoe UI Semibold', 9), state='normal', **kwargs):
        super().__init__(parent, width=width, height=height, highlightthickness=0, bd=0,
                         bg=parent.cget('bg'), cursor='hand2' if state=='normal' else '', **kwargs)
        self.text=text; self.command=command; self.radius=radius
        self.normal_bg=bg; self.hover_bg=hover; self.fg=fg
        self.disabled_bg=disabled_bg; self.disabled_fg=disabled_fg
        self.font=font; self._state=state
        self._draw()
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        self.bind('<Button-1>', self._on_click)
        self.bind('<Configure>', lambda e: self._draw(False))

    def _draw(self, hover=False):
        self.delete('all')
        w=max(2, self.winfo_width()); h=max(2, self.winfo_height())
        if self._state == 'disabled':
            bg=self.disabled_bg; fg=self.disabled_fg
        else:
            bg=self.hover_bg if hover else self.normal_bg; fg=self.fg
        _rounded_rect(self, 1, 1, w-1, h-1, self.radius, fill=bg, outline=bg)
        self.create_text(w/2, h/2, text=self.text, fill=fg, font=self.font)

    def _on_enter(self, _):
        if self._state!='disabled': self._draw(True)
    def _on_leave(self, _):
        self._draw(False)
    def _on_click(self, _):
        if self._state!='disabled' and self.command:
            self.command()

    def configure(self, cnf=None, **kwargs):
        if cnf:
            kwargs.update(cnf)
        if 'state' in kwargs:
            self._state=kwargs.pop('state')
            self.configure(cursor='hand2' if self._state=='normal' else '')
        if 'text' in kwargs:
            self.text=kwargs.pop('text')
        if 'width' in kwargs:
            super().configure(width=kwargs.pop('width'))
        if 'height' in kwargs:
            super().configure(height=kwargs.pop('height'))
        if kwargs:
            super().configure(**kwargs)
        self._draw(False)
    config = configure

    def set_palette(self, bg, hover, fg):
        self.normal_bg=bg; self.hover_bg=hover; self.fg=fg; self._draw(False)


class ModeTab(RoundedButton):
    def __init__(self, parent, text, command, selected=False, **kwargs):
        self.selected = selected
        super().__init__(parent, text=text, command=command, width=210 if selected else 170,
                         height=48 if selected else 40, radius=16,
                         bg='#6D5DFB' if selected else '#EEF0F7',
                         hover='#5949E8' if selected else '#E3E6F0',
                         fg='#FFFFFF' if selected else '#667085',
                         font=('Segoe UI Semibold', 10 if selected else 9), **kwargs)

    def set_selected(self, selected):
        self.selected=selected
        self.font=('Segoe UI Semibold', 10 if selected else 9)
        self.configure(width=210 if selected else 170, height=48 if selected else 40)
        self.set_palette('#6D5DFB' if selected else '#EEF0F7',
                         '#5949E8' if selected else '#E3E6F0',
                         '#FFFFFF' if selected else '#667085')


class HelpIcon(tk.Label):
    """Contextual help icon with hover, click, and keyboard support."""
    def __init__(self, parent, text, **kwargs):
        super().__init__(
            parent, text="?", bg="#EEF0FF", fg="#6D5DFB",
            font=("Segoe UI Semibold", 8), width=2,
            cursor="hand2", takefocus=True, **kwargs
        )
        self.help_text = text
        self.popup = None
        self.bind("<Enter>", self._show)
        self.bind("<Leave>", self._hide)
        self.bind("<Button-1>", self._toggle)
        self.bind("<FocusIn>", self._show)
        self.bind("<FocusOut>", self._hide)
        self.bind("<Return>", self._toggle)
        self.bind("<space>", self._toggle)

    def _show(self, event=None):
        if self.popup is not None and self.popup.winfo_exists():
            return
        self.update_idletasks()
        x = self.winfo_rootx() + self.winfo_width() + 6
        y = self.winfo_rooty() + self.winfo_height() + 2
        self.popup = tk.Toplevel(self)
        self.popup.wm_overrideredirect(True)
        self.popup.wm_geometry(f"+{x}+{y}")
        frame = tk.Frame(
            self.popup, bg="#FFFFFF",
            highlightbackground="#D8DCE8", highlightthickness=1
        )
        frame.pack()
        tk.Label(
            frame, text=self.help_text, bg="#FFFFFF", fg="#172033",
            font=("TH Sarabun New", 14), justify="left", wraplength=420,
            padx=12, pady=10
        ).pack()

    def _hide(self, event=None):
        if self.popup is not None:
            try:
                self.popup.destroy()
            except tk.TclError:
                pass
            self.popup = None

    def _toggle(self, event=None):
        if self.popup is not None and self.popup.winfo_exists():
            self._hide()
        else:
            self._show()


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DETP_FILE = (
    BASE_DIR
    / "output"
    / "detp_core_v1.npy"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

FLOW_TO_ALPHA_DB_SCALE = 1e-6

TAU_US = np.arange(
    0,
    501,
    dtype=float,
)

TAU_SECONDS = (
    TAU_US
    * 1e-6
)


# ============================================================
# HELPERS
# ============================================================

def format_flow_value(value):
    """
    Display flow values cleanly.

    Example:
        5.0  -> "5"
        2.5  -> "2.5"
    """

    value = float(value)

    if value.is_integer():
        return str(int(value))

    return f"{value:g}"


# ============================================================
# LOAD AVAILABLE REFERENCE CONDITIONS
# ============================================================

def load_reference_conditions():
    """
    Read all unique commanded shallow/deep flow combinations
    available in the physical reference MAT file.

    Dataset_set_intra40:
        row 0 = commanded shallow flow
        row 1 = commanded deep flow
    """

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            "Reference dataset not found:\n"
            f"{REFERENCE_FILE}"
        )

    data = loadmat(
        REFERENCE_FILE
    )

    if "Dataset_set_intra40" not in data:
        raise KeyError(
            "Dataset_set_intra40 not found "
            "in reference MAT file."
        )

    dataset = data[
        "Dataset_set_intra40"
    ]

    if (
        dataset.ndim != 2
        or dataset.shape[0] < 2
    ):
        raise ValueError(
            "Unexpected Dataset_set_intra40 shape: "
            f"{dataset.shape}"
        )

    shallow_values = dataset[0]
    deep_values = dataset[1]

    conditions = {}

    for shallow, deep in zip(
        shallow_values,
        deep_values,
    ):

        shallow = float(shallow)
        deep = float(deep)

        if shallow not in conditions:
            conditions[shallow] = set()

        conditions[shallow].add(
            deep
        )

    # Sort shallow and deep values
    sorted_conditions = {}

    for shallow in sorted(
        conditions.keys()
    ):

        sorted_conditions[shallow] = sorted(
            conditions[shallow]
        )

    return sorted_conditions


# ============================================================
# FLOW -> alphaDb
# ============================================================

def flow_to_alpha_db(flow):
    """
    Temporary Core V1 mapping.

        alphaDb = flow * 1e-6 mm^2/s

    This function will be replaced after final calibration.
    """

    flow = float(flow)

    if not np.isfinite(flow):
        raise ValueError(
            "Flow must be finite."
        )

    if flow < 0:
        raise ValueError(
            "Flow must be >= 0."
        )

    return (
        flow
        * FLOW_TO_ALPHA_DB_SCALE
    )


# ============================================================
# DYNAMIC g1
# ============================================================

def calculate_dynamic_g1(
    detp,
    tau_seconds,
    detector_id,
    alpha_db_shallow,
    alpha_db_deep,
):
    """
    Calculate absorption-weighted normalized g1(tau)
    for one detector.
    """

    photon_data = extract_photon_data(
        detp
    )

    detector_ids = (
        photon_data[
            "detector_id"
        ]
    )

    mask = (
        detector_ids
        == detector_id
    )

    if not np.any(mask):
        raise ValueError(
            f"No photons detected at detector "
            f"{detector_id}"
        )

    path_shallow = (
        photon_data[
            "path_shallow"
        ][mask]
    )

    path_deep = (
        photon_data[
            "path_deep"
        ][mask]
    )

    momentum_shallow = (
        photon_data[
            "momentum_shallow"
        ][mask]
    )

    momentum_deep = (
        photon_data[
            "momentum_deep"
        ][mask]
    )

    # --------------------------------------------------------
    # Absorption weighting
    # --------------------------------------------------------

    absorption_weight = np.exp(
        -MU_A_MM
        * (
            path_shallow
            + path_deep
        )
    )

    weight_sum = np.sum(
        absorption_weight
    )

    if (
        not np.isfinite(weight_sum)
        or weight_sum <= 0
    ):
        raise ValueError(
            "Invalid absorption weights."
        )

    # --------------------------------------------------------
    # Dynamic decay
    # --------------------------------------------------------

    decay_rate = (
        2.0
        * K0**2
        * (
            momentum_shallow
            * alpha_db_shallow
            +
            momentum_deep
            * alpha_db_deep
        )
    )

    g1 = np.empty(
        len(tau_seconds),
        dtype=float,
    )

    for index, tau in enumerate(
        tau_seconds
    ):

        dynamic_term = np.exp(
            -decay_rate
            * tau
        )

        g1[index] = (
            np.sum(
                absorption_weight
                * dynamic_term
            )
            / weight_sum
        )

    g1 = (
        g1
        / g1[0]
    )

    if not np.all(
        np.isfinite(g1)
    ):
        raise ValueError(
            "Generated g1 contains "
            "non-finite values."
        )

    return g1


# ============================================================
# RUN VALIDATION
# ============================================================

def run_validation(
    shallow_flow,
    deep_flow,
):
    """
    Compare:
      1) Proposed model (this project)
      2) Baseline pathlength model
      3) Physical reference
    for one known reference condition.
    """

    shallow_flow = float(shallow_flow)
    deep_flow = float(deep_flow)

    alpha_db_shallow = flow_to_alpha_db(shallow_flow)
    alpha_db_deep = flow_to_alpha_db(deep_flow)

    if not DETP_FILE.exists():
        raise FileNotFoundError(
            "Detected-photon file not found:\n"
            f"{DETP_FILE}\n\n"
            "Run simulation.py first."
        )

    detp = np.load(DETP_FILE)

    if detp.ndim != 2 or detp.shape[0] != 5:
        raise ValueError(
            "Expected detp shape (5, N). "
            f"Got {detp.shape}"
        )

    reference = load_reference_condition(
        shallow_flow=shallow_flow,
        deep_flow=deep_flow,
    )

    baseline_paths = load_baseline_pathlengths()
    results = {}
    detector_ids = detp[0].astype(int)

    all_proposed_errors = []
    all_baseline_errors = []

    for sds_cm in (1, 2, 3):
        reference_g2 = reference["g2"][sds_cm]["mean"]
        beta = float(reference_g2[0] - 1.0)

        # Proposed model
        g1 = calculate_dynamic_g1(
            detp=detp,
            tau_seconds=TAU_SECONDS,
            detector_id=sds_cm,
            alpha_db_shallow=alpha_db_shallow,
            alpha_db_deep=alpha_db_deep,
        )

        generated_g2 = calculate_g2(
            g1=g1,
            beta=beta,
        )

        # Baseline model from the comparison code
        Ls, Ld = baseline_paths[sds_cm]

        baseline_g2 = calculate_baseline_g2(
            Ls=Ls,
            Ld=Ld,
            flow_s=shallow_flow,
            flow_d=deep_flow,
            tau_array=TAU_SECONDS,
            beta=beta,
        )

        proposed_error = generated_g2 - reference_g2
        baseline_error = baseline_g2 - reference_g2

        proposed_mae = float(np.mean(np.abs(proposed_error)))
        proposed_rmse = float(np.sqrt(np.mean(proposed_error ** 2)))
        proposed_max_error = float(np.max(np.abs(proposed_error)))

        baseline_mae = float(np.mean(np.abs(baseline_error)))
        baseline_rmse = float(np.sqrt(np.mean(baseline_error ** 2)))
        baseline_max_error = float(np.max(np.abs(baseline_error)))

        all_proposed_errors.append(proposed_error)
        all_baseline_errors.append(baseline_error)

        if np.isclose(proposed_rmse, baseline_rmse, rtol=1e-6, atol=1e-12):
            better = "Tie"
        elif proposed_rmse < baseline_rmse:
            better = "Proposed"
        else:
            better = "Baseline"

        photon_count = int(np.sum(detector_ids == sds_cm))

        results[sds_cm] = {
            "beta": beta,
            "g1": g1,
            "generated_g2": generated_g2,
            "baseline_g2": baseline_g2,
            "reference_g2": reference_g2,
            "mae": proposed_mae,
            "rmse": proposed_rmse,
            "max_error": proposed_max_error,
            "baseline_mae": baseline_mae,
            "baseline_rmse": baseline_rmse,
            "baseline_max_error": baseline_max_error,
            "better": better,
            "photons": photon_count,
        }

    proposed_all = np.concatenate(all_proposed_errors)
    baseline_all = np.concatenate(all_baseline_errors)

    proposed_combined_rmse = float(
        np.sqrt(np.mean(proposed_all ** 2))
    )
    baseline_combined_rmse = float(
        np.sqrt(np.mean(baseline_all ** 2))
    )

    if np.isclose(
        proposed_combined_rmse,
        baseline_combined_rmse,
        rtol=1e-6,
        atol=1e-12,
    ):
        overall_better = "Tie"
        verdict = (
            "Result: Proposed and Baseline models have "
            "approximately equal combined RMSE for this condition."
        )
    elif proposed_combined_rmse < baseline_combined_rmse:
        overall_better = "Proposed"
        improvement = (
            (baseline_combined_rmse - proposed_combined_rmse)
            / baseline_combined_rmse
            * 100.0
        )
        verdict = (
            "Result: Proposed model has lower combined RMSE "
            f"for this condition ({improvement:.2f}% lower than Baseline)."
        )
    else:
        overall_better = "Baseline"
        improvement = (
            (proposed_combined_rmse - baseline_combined_rmse)
            / proposed_combined_rmse
            * 100.0
        )
        verdict = (
            "Result: Baseline model has lower combined RMSE "
            f"for this condition ({improvement:.2f}% lower than Proposed)."
        )

    return {
        "detp": detp,
        "reference": reference,
        "results": results,
        "shallow_flow": shallow_flow,
        "deep_flow": deep_flow,
        "alpha_db_shallow": alpha_db_shallow,
        "alpha_db_deep": alpha_db_deep,
        "proposed_combined_rmse": proposed_combined_rmse,
        "baseline_combined_rmse": baseline_combined_rmse,
        "overall_better": overall_better,
        "verdict": verdict,
    }


# ============================================================
# GUI
# ============================================================

class DCSSimulatorGUI:

    # --------------------------------------------------------
    # VISUAL SYSTEM
    # --------------------------------------------------------
    COLORS = {
        "bg": "#F6F7FB",
        "surface": "#FFFFFF",
        "surface_soft": "#F2F4FF",
        "navy": "#111827",
        "navy_2": "#1F2937",
        "text": "#172033",
        "muted": "#667085",
        "line": "#E6E8F0",
        "violet": "#6D5DFB",
        "violet_hover": "#5949E8",
        "teal": "#16B8A6",
        "teal_soft": "#E8FBF7",
        "sky": "#3B82F6",
        "orange": "#F59E0B",
        "danger": "#EF4444",
        "success": "#10B981",
        "disabled": "#D6DAE5",
        "disabled_text": "#98A2B3",
    }

    def __init__(self, root):
        self.root = root
        self.root.title("DCS Simulator")
        self.root.geometry("1480x900")
        self.root.minsize(1180, 740)
        self.root.configure(bg=self.COLORS["bg"])

        self.last_validation_output = None
        self.last_human_output = None
        self.human_dataset = None

        self._configure_styles()

        try:
            self.reference_conditions = load_reference_conditions()
        except Exception as error:
            messagebox.showerror("Reference Dataset Error", str(error))
            self.reference_conditions = {}

        # Top application bar
        appbar = tk.Frame(root, bg=self.COLORS["navy"], height=96, highlightthickness=0)
        appbar.pack(fill="x")
        appbar.pack_propagate(False)

        brand = tk.Frame(appbar, bg=self.COLORS["navy"])
        brand.pack(side="left", fill="y", padx=24, pady=(10, 8))

        tk.Label(brand, text="DCS SIMULATOR", bg=self.COLORS["navy"], fg="#FFFFFF",
                 font=("Segoe UI Semibold", 22)).pack(anchor="w")
        tk.Label(brand, text="Validation  •  Synthetic Human Dataset Generation",
                 bg=self.COLORS["navy"], fg="#B9C0D4", font=("Segoe UI", 10)).pack(anchor="w", pady=(4, 0))

        badge = tk.Label(appbar, text="DCS / MCX", bg=self.COLORS["teal"], fg="#FFFFFF",
                         font=("Segoe UI Semibold", 10), padx=14, pady=8)
        badge.pack(side="right", padx=24, pady=25)

        # Custom mode navigation. The selected mode is intentionally larger.
        nav = tk.Frame(root, bg=self.COLORS["bg"], height=68)
        nav.pack(fill="x", padx=20, pady=(12, 0))
        nav.pack_propagate(False)
        self.mode_slots=[]
        for _ in range(2):
            slot=tk.Frame(nav, bg=self.COLORS["bg"], width=230, height=58)
            slot.pack(side="left", padx=(0, 10))
            slot.pack_propagate(False)
            self.mode_slots.append(slot)

        # Notebook content only; native square tabs are hidden.
        self.notebook = ttk.Notebook(root, style="Hidden.TNotebook")
        self.notebook.pack(fill="both", expand=True, padx=20, pady=(0, 18))
        self.validation_tab = tk.Frame(self.notebook, bg=self.COLORS["bg"])
        self.human_tab = tk.Frame(self.notebook, bg=self.COLORS["bg"])
        self.notebook.add(self.validation_tab, text="Model Validation")
        self.notebook.add(self.human_tab, text="Human Dataset Generator")

        self.validation_mode_button = ModeTab(self.mode_slots[0], "Model Validation", lambda: self._switch_mode(0), selected=False)
        self.human_mode_button = ModeTab(self.mode_slots[1], "Human Dataset Generator", lambda: self._switch_mode(1), selected=True)
        self.validation_mode_button.place(relx=.5, rely=.5, anchor='center')
        self.human_mode_button.place(relx=.5, rely=.5, anchor='center')

        self.build_validation_tab()
        self.build_human_tab()
        self._switch_mode(1)

    # ========================================================
    # STYLE / COMPONENT HELPERS
    # ========================================================

    def _configure_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        c = self.COLORS

        style.configure("Hidden.TNotebook", background=c["bg"], borderwidth=0)
        style.layout("Hidden.TNotebook.Tab", [])
        style.configure(
            "Modern.TNotebook",
            background=c["bg"],
            borderwidth=0,
            tabmargins=(0, 0, 0, 0),
        )
        style.configure(
            "Modern.TNotebook.Tab",
            font=("Segoe UI Semibold", 10),
            padding=(18, 10),
            background="#E9ECF5",
            foreground=c["muted"],
            borderwidth=0,
        )
        style.map(
            "Modern.TNotebook.Tab",
            background=[("selected", c["violet"])],
            foreground=[("selected", "#FFFFFF")],
        )

        style.configure(
            "Modern.TCombobox",
            padding=7,
            fieldbackground="#FFFFFF",
            background="#FFFFFF",
            foreground=c["text"],
            bordercolor="#D8DCE8",
            lightcolor="#D8DCE8",
            darkcolor="#D8DCE8",
            arrowsize=14,
        )
        style.map(
            "Modern.TCombobox",
            bordercolor=[("focus", c["violet"])],
            lightcolor=[("focus", c["violet"])],
            darkcolor=[("focus", c["violet"])],
        )

        style.configure(
            "Modern.TEntry",
            padding=8,
            fieldbackground="#FFFFFF",
            foreground=c["text"],
            bordercolor="#D8DCE8",
            lightcolor="#D8DCE8",
            darkcolor="#D8DCE8",
        )
        style.map(
            "Modern.TEntry",
            bordercolor=[("focus", c["violet"])],
            lightcolor=[("focus", c["violet"])],
            darkcolor=[("focus", c["violet"])],
        )

        style.configure(
            "Modern.Treeview",
            background="#FFFFFF",
            fieldbackground="#FFFFFF",
            foreground=c["text"],
            rowheight=29,
            borderwidth=0,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Modern.Treeview.Heading",
            background="#EEF0FF",
            foreground=c["text"],
            font=("Segoe UI Semibold", 9),
            padding=(7, 8),
            borderwidth=0,
        )
        style.map(
            "Modern.Treeview",
            background=[("selected", "#E8E5FF")],
            foreground=[("selected", c["navy"])],
        )

    def _card(self, parent, padx=18, pady=16):
        outer = RoundedCard(
            parent,
            bg_color=self.COLORS["surface"],
            border_color=self.COLORS["line"],
            radius=18,
        )
        inner = outer.inner
        inner.pack_propagate(True)
        # Padding is provided by an inner content frame so rounded corners stay visible.
        content = tk.Frame(inner, bg=self.COLORS["surface"])
        content.pack(fill="both", expand=True, padx=padx, pady=pady)
        return outer, content

    def _eyebrow(self, parent, text, color=None):
        tk.Label(
            parent,
            text=text.upper(),
            bg=parent.cget("bg"),
            fg=color or self.COLORS["violet"],
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w")

    def _title(self, parent, text, size=18):
        tk.Label(
            parent,
            text=text,
            bg=parent.cget("bg"),
            fg=self.COLORS["text"],
            font=("Segoe UI Semibold", size),
        ).pack(anchor="w")

    def _body(self, parent, text, wrap=320, pady=(4, 0)):
        tk.Label(
            parent,
            text=text,
            bg=parent.cget("bg"),
            fg=self.COLORS["muted"],
            font=("Segoe UI", 9),
            wraplength=wrap,
            justify="left",
        ).pack(anchor="w", pady=pady)

    def _field_label(self, parent, text):
        return tk.Label(
            parent,
            text=text,
            bg=parent.cget("bg"),
            fg=self.COLORS["text"],
            font=("Segoe UI Semibold", 9),
        )

    def _helper_label(self, parent, text):
        return tk.Label(
            parent,
            text=text,
            bg=parent.cget("bg"),
            fg=self.COLORS["muted"],
            font=("Segoe UI", 8),
        )

    def _button(self, parent, text, command, kind="primary", state="normal"):
        palette = {
            "primary": (self.COLORS["violet"], self.COLORS["violet_hover"], "#FFFFFF"),
            "teal": (self.COLORS["teal"], "#0E9E90", "#FFFFFF"),
            "dark": (self.COLORS["navy_2"], self.COLORS["navy"], "#FFFFFF"),
            "soft": ("#EEF0FF", "#E0E3FF", self.COLORS["violet"]),
        }
        bg, active, fg = palette[kind]
        return RoundedButton(
            parent, text=text, command=command, state=state,
            width=220, height=42, radius=14, bg=bg, hover=active, fg=fg,
            disabled_bg="#EEF0F4", disabled_fg=self.COLORS["disabled_text"],
            font=("Segoe UI Semibold", 9),
        )

    def _switch_mode(self, index):
        self.notebook.select(index)
        self.validation_mode_button.set_selected(index == 0)
        self.human_mode_button.set_selected(index == 1)

    def _set_status_badge(self, label, text, kind="ready"):
        palette = {
            "ready": ("#EEF0FF", self.COLORS["violet"]),
            "processing": ("#FFF7E8", self.COLORS["orange"]),
            "success": (self.COLORS["teal_soft"], self.COLORS["success"]),
            "error": ("#FFF0F0", self.COLORS["danger"]),
        }
        bg, fg = palette.get(kind, palette["ready"])
        label.configure(text=text, bg=bg, fg=fg)

    # ========================================================
    # TAB 1 — VALIDATION
    # ========================================================

    def build_validation_tab(self):
        c = self.COLORS
        self.validation_tab.grid_columnconfigure(0, weight=0)
        self.validation_tab.grid_columnconfigure(1, weight=1)
        self.validation_tab.grid_rowconfigure(0, weight=1)

        # LEFT COLUMN
        left_wrap = tk.Frame(self.validation_tab, bg=c["bg"], width=340)
        left_wrap.grid(row=0, column=0, sticky="ns", padx=(0, 16))
        left_wrap.grid_propagate(False)

        intro_outer, intro = self._card(left_wrap, padx=20, pady=18)
        intro_outer.pack(fill="x")
        self._eyebrow(intro, "Reference Validation", c["teal"])
        self._title(intro, "Model Validation", 19)
        self._body(
            intro,
            "Compare the proposed DCS model with a baseline model and physical reference measurements.",
            wrap=285,
        )

        cond_outer, cond = self._card(left_wrap, padx=18, pady=16)
        cond_outer.pack(fill="x", pady=(14, 0))
        self._title(cond, "Reference Condition", 13)
        self._body(cond, "Choose only flow pairs available in the physical dataset.", wrap=285, pady=(3, 13))

        self.validation_shallow_var = tk.StringVar()
        self.validation_deep_var = tk.StringVar()

        self._field_label(cond, "Shallow Flow Rate").pack(anchor="w")
        shallow_values = [format_flow_value(v) for v in self.reference_conditions.keys()]
        self.validation_shallow_combo = ttk.Combobox(
            cond,
            textvariable=self.validation_shallow_var,
            values=shallow_values,
            state="readonly",
            style="Modern.TCombobox",
        )
        self.validation_shallow_combo.pack(fill="x", pady=(5, 2))
        self.validation_shallow_combo.set("Select shallow flow")
        self.validation_shallow_combo.bind("<<ComboboxSelected>>", self.on_shallow_changed)
        self._helper_label(cond, "ml/min/100g").pack(anchor="w", pady=(0, 12))

        self._field_label(cond, "Deep Flow Rate").pack(anchor="w")
        self.validation_deep_combo = ttk.Combobox(
            cond,
            textvariable=self.validation_deep_var,
            state="readonly",
            style="Modern.TCombobox",
        )
        self.validation_deep_combo.pack(fill="x", pady=(5, 2))
        self.validation_deep_combo.set("Select shallow first")
        self._helper_label(cond, "ml/min/100g").pack(anchor="w", pady=(0, 13))

        self.validation_run_button = self._button(
            cond,
            "Generate & Compare",
            self.run_reference_validation,
            kind="teal",
        )
        self.validation_run_button.pack(fill="x")

        status_outer, status = self._card(left_wrap, padx=18, pady=15)
        status_outer.pack(fill="x", pady=(14, 0))
        top_status = tk.Frame(status, bg=c["surface"])
        top_status.pack(fill="x")
        self._title(top_status, "Validation Status", 13)

        self.validation_status_badge = tk.Label(
            status,
            text="Ready",
            bg="#EEF0FF",
            fg=c["violet"],
            font=("Segoe UI Semibold", 9),
            padx=10,
            pady=5,
        )
        self.validation_status_badge.pack(anchor="w", pady=(10, 10))

        self.validation_status_var = tk.StringVar(value="Ready")
        self.validation_condition_var = tk.StringVar(value="Condition: -")
        self.validation_alpha_var = tk.StringVar(value="alphaDb: -")
        self.validation_photon_var = tk.StringVar(value="Detected photons: -")

        for var in (self.validation_condition_var, self.validation_alpha_var, self.validation_photon_var):
            tk.Label(
                status,
                textvariable=var,
                bg=c["surface"],
                fg=c["muted"],
                font=("Segoe UI", 8),
                justify="left",
                wraplength=285,
            ).pack(anchor="w", pady=(0, 4))

        # RIGHT COLUMN
        right = tk.Frame(self.validation_tab, bg=c["bg"])
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=3)
        right.grid_rowconfigure(1, weight=1)

        graph_outer, graph_card = self._card(right, padx=16, pady=14)
        graph_outer.grid(row=0, column=0, sticky="nsew")
        graph_card.grid_columnconfigure(0, weight=1)
        graph_card.grid_rowconfigure(1, weight=1)

        graph_head = tk.Frame(graph_card, bg=c["surface"])
        graph_head.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        graph_head.grid_columnconfigure(0, weight=1)

        tk.Label(
            graph_head,
            text="Proposed vs Baseline vs Physical Reference",
            bg=c["surface"],
            fg=c["text"],
            font=("Segoe UI Semibold", 14),
        ).grid(row=0, column=0, sticky="w")

        tk.Label(
            graph_head,
            text="Graph View",
            bg=c["surface"],
            fg=c["muted"],
            font=("Segoe UI", 8),
        ).grid(row=0, column=1, padx=(8, 6))

        self.sds_view_var = tk.StringVar(value="1 cm")
        self.sds_view_combo = ttk.Combobox(
            graph_head,
            textvariable=self.sds_view_var,
            values=["1 cm", "2 cm", "3 cm", "All"],
            state="readonly",
            width=8,
            style="Modern.TCombobox",
        )
        self.sds_view_combo.grid(row=0, column=2)
        self.sds_view_combo.bind("<<ComboboxSelected>>", self.on_sds_view_changed)

        self.validation_figure = Figure(figsize=(9, 5), dpi=100, facecolor=c["surface"])
        self.validation_axis = self.validation_figure.add_subplot(111)
        self._style_validation_axis("Select a reference condition")

        self.validation_canvas = FigureCanvasTkAgg(self.validation_figure, master=graph_card)
        self.validation_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        metrics_outer, metrics = self._card(right, padx=16, pady=12)
        metrics_outer.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        self._title(metrics, "Validation Metrics", 13)

        columns = ("sds", "proposed_rmse", "baseline_rmse", "better")
        self.validation_metrics_table = ttk.Treeview(
            metrics,
            columns=columns,
            show="headings",
            height=4,
            style="Modern.Treeview",
        )
        headings = {
            "sds": "SDS",
            "proposed_rmse": "Proposed RMSE",
            "baseline_rmse": "Baseline RMSE",
            "better": "Better Match",
        }
        for column in columns:
            self.validation_metrics_table.heading(column, text=headings[column])
            self.validation_metrics_table.column(column, width=120, anchor="center")
        self.validation_metrics_table.pack(fill="x", pady=(9, 6))

        self.validation_verdict_var = tk.StringVar(
            value="Run a condition to compare Proposed and Baseline models."
        )
        tk.Label(
            metrics,
            textvariable=self.validation_verdict_var,
            bg=c["surface"],
            fg=c["navy_2"],
            font=("Segoe UI Semibold", 9),
            wraplength=950,
            justify="left",
        ).pack(anchor="w")

    def _style_validation_axis(self, title=None):
        c = self.COLORS
        ax = self.validation_axis
        ax.set_facecolor("#FBFCFF")
        ax.set_xlabel("Delay time τ (μs)", color=c["muted"])
        ax.set_ylabel("g₂(τ)", color=c["muted"])
        if title is not None:
            ax.set_title(title, color=c["text"], fontsize=12, pad=12)
        ax.grid(True, color="#E8EAF2", linewidth=0.8)
        ax.tick_params(colors=c["muted"], labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#E0E3EC")

    def on_shallow_changed(
        self,
        event=None,
    ):

        self.update_deep_dropdown()


    def update_deep_dropdown(
        self,
    ):

        shallow_text = (
            self.validation_shallow_var.get()
        )

        if not shallow_text:
            return

        shallow = float(
            shallow_text
        )

        # Find exact numeric key
        matched_key = None

        for key in (
            self.reference_conditions.keys()
        ):

            if np.isclose(
                key,
                shallow,
            ):
                matched_key = key
                break

        if matched_key is None:

            self.validation_deep_combo[
                "values"
            ] = []

            self.validation_deep_var.set(
                ""
            )

            return

        deep_values = [
            format_flow_value(value)
            for value
            in self.reference_conditions[
                matched_key
            ]
        ]

        self.validation_deep_combo[
            "values"
        ] = deep_values

        if deep_values:

            # Prefer 20 when available
            if "20" in deep_values:
                selected = "20"
            else:
                selected = deep_values[0]

            self.validation_deep_var.set(
                selected
            )

        else:

            self.validation_deep_var.set(
                ""
            )


    # ========================================================
    # RUN REFERENCE VALIDATION
    # ========================================================

    def run_reference_validation(
        self,
    ):

        try:

            shallow_text = (
                self.validation_shallow_var.get()
            )

            deep_text = (
                self.validation_deep_var.get()
            )

            if (
                not shallow_text
                or not deep_text
            ):
                raise ValueError(
                    "Select both shallow "
                    "and deep flow."
                )

            shallow_flow = float(
                shallow_text
            )

            deep_flow = float(
                deep_text
            )

            self.validation_run_button.config(state="disabled")

            self.validation_status_var.set("Processing...")
            self._set_status_badge(self.validation_status_badge, "Processing...", "processing")

            self.root.update_idletasks()

            output = run_validation(
                shallow_flow=shallow_flow,
                deep_flow=deep_flow,
            )

            results = (
                output["results"]
            )

            self.last_validation_output = output

            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            self.validation_condition_var.set(
                "Condition: "
                f"{shallow_flow:g} / "
                f"{deep_flow:g}"
            )

            self.validation_alpha_var.set(
                "alphaDb: "
                f"{output['alpha_db_shallow']:.3e} / "
                f"{output['alpha_db_deep']:.3e} mm²/s"
            )

            self.validation_photon_var.set(
                "Detected photons: "
                f"{output['detp'].shape[1]:,}"
            )

            # ------------------------------------------------
            # Graph
            # ------------------------------------------------

            self.update_validation_graph(
                results,
                shallow_flow,
                deep_flow,
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            self.update_validation_metrics(
                results,
                output,
            )

            self.validation_status_var.set("Complete")
            self._set_status_badge(self.validation_status_badge, "Complete", "success")

        except Exception as error:

            self.validation_status_var.set("Error")
            self._set_status_badge(self.validation_status_badge, "Error", "error")

            messagebox.showerror(
                "Validation Error",
                str(error),
            )

        finally:

            self.validation_run_button.config(state="normal")


    # ========================================================
    # VALIDATION GRAPH
    # ========================================================

    def on_sds_view_changed(
        self,
        event=None,
    ):
        """
        Redraw the graph immediately using the latest validation
        result. No simulation rerun is required.
        """
        if self.last_validation_output is None:
            return

        output = self.last_validation_output

        self.update_validation_graph(
            output["results"],
            output["shallow_flow"],
            output["deep_flow"],
        )


    def update_validation_graph(
        self,
        results,
        shallow_flow,
        deep_flow,
    ):

        self.validation_axis.clear()

        view = self.sds_view_var.get()

        if view == "All":
            selected_sds = (1, 2, 3)
        else:
            selected_sds = (int(view.split()[0]),)

        for sds_cm in selected_sds:
            generated = results[sds_cm]["generated_g2"]
            baseline = results[sds_cm]["baseline_g2"]
            reference = results[sds_cm]["reference_g2"]

            self.validation_axis.plot(
                TAU_US,
                generated,
                linewidth=2.0,
                label=f"Proposed SDS {sds_cm} cm",
            )

            self.validation_axis.plot(
                TAU_US,
                baseline,
                linestyle=":",
                linewidth=2.0,
                label=f"Baseline SDS {sds_cm} cm",
            )

            self.validation_axis.plot(
                TAU_US,
                reference,
                linestyle="--",
                linewidth=2.0,
                label=f"Physical Reference SDS {sds_cm} cm",
            )

        self.validation_axis.set_xlabel(
            "Delay time τ (μs)"
        )

        self.validation_axis.set_ylabel(
            "g₂(τ)"
        )

        if view == "All":
            view_title = "All SDS"
        else:
            view_title = f"SDS {view}"

        self.validation_axis.set_title(
            f"{view_title}\\n"
            f"Shallow = {shallow_flow:g}, "
            f"Deep = {deep_flow:g}"
        )

        self.validation_axis.grid(True, color="#E8EAF2", linewidth=0.8)
        self.validation_axis.tick_params(colors=self.COLORS["muted"], labelsize=9)
        for spine in self.validation_axis.spines.values():
            spine.set_color("#E0E3EC")

        self.validation_axis.legend(
            fontsize=8,
            ncol=1 if view != "All" else 3,
            frameon=False,
        )

        self.validation_figure.tight_layout()
        self.validation_canvas.draw()


    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    def update_validation_metrics(
        self,
        results,
        output,
    ):

        for item in self.validation_metrics_table.get_children():
            self.validation_metrics_table.delete(item)

        for sds_cm in (1, 2, 3):
            data = results[sds_cm]

            self.validation_metrics_table.insert(
                "",
                "end",
                values=(
                    f"{sds_cm} cm",
                    f"{data['rmse']:.6f}",
                    f"{data['baseline_rmse']:.6f}",
                    (
                        f"{data['better']} ✓"
                        if data["better"] != "Tie"
                        else "Tie"
                    ),
                ),
            )

        self.validation_metrics_table.insert(
            "",
            "end",
            values=(
                "Combined",
                f"{output['proposed_combined_rmse']:.6f}",
                f"{output['baseline_combined_rmse']:.6f}",
                (
                    f"{output['overall_better']} ✓"
                    if output["overall_better"] != "Tie"
                    else "Tie"
                ),
            ),
        )

        proposed = output["proposed_combined_rmse"]
        baseline = output["baseline_combined_rmse"]

        if output["overall_better"] == "Proposed":
            percent = (
                (baseline - proposed)
                / baseline
                * 100.0
            )
            verdict = (
                "✓ Overall: Proposed model is the better match for this condition.  "
                f"Combined RMSE: {proposed:.6f} vs {baseline:.6f}.  "
                f"Proposed RMSE is {percent:.2f}% lower than Baseline."
            )
        elif output["overall_better"] == "Baseline":
            percent = (
                (proposed - baseline)
                / proposed
                * 100.0
            )
            verdict = (
                "✓ Overall: Baseline model is the better match for this condition.  "
                f"Combined RMSE: {baseline:.6f} vs {proposed:.6f}.  "
                f"Baseline RMSE is {percent:.2f}% lower than Proposed."
            )
        else:
            verdict = (
                "Overall: Both models have approximately equal combined RMSE "
                "for this condition."
            )

        self.validation_verdict_var.set(verdict)


    # ========================================================
    # TAB 2
    # ========================================================


    # ========================================================
    # TAB 2 — HUMAN DATASET GENERATOR
    # ========================================================

    def build_human_tab(self):
        c = self.COLORS
        self.human_tab.grid_columnconfigure(0, weight=0)
        self.human_tab.grid_columnconfigure(1, weight=1)
        self.human_tab.grid_rowconfigure(0, weight=1)

        left_wrap = tk.Frame(self.human_tab, bg=c["bg"], width=360)
        left_wrap.grid(row=0, column=0, sticky="ns", padx=(0, 16))
        left_wrap.grid_propagate(False)

        intro_outer, intro = self._card(left_wrap, padx=20, pady=18)
        intro_outer.pack(fill="x")
        self._eyebrow(intro, "Synthetic DCS", c["violet"])
        self._title(intro, "Human Dataset Generator", 19)
        self._body(
            intro,
            "Generate subject-specific synthetic DCS g₂ datasets from a calibrated digital phantom.",
            wrap=300,
        )

        # Validated flow: inputs + generation in ONE card
        gen_outer, subject = self._card(left_wrap, padx=18, pady=16)
        gen_outer.pack(fill="x", pady=(14, 0))
        self._title(subject, "Subject & Dataset Generation", 13)
        self._body(
            subject,
            "Enter subject measurements, then generate the synthetic flow grid.",
            wrap=300,
            pady=(3, 13),
        )

        self.sex_var = tk.StringVar()
        self.sat_var = tk.StringVar()
        self.skin_var = tk.StringVar()
        self.human_placeholder_entries = {}

        self._field_label(subject, "Sex *").pack(anchor="w")
        self.sex_combo = ttk.Combobox(
            subject,
            textvariable=self.sex_var,
            values=["Female", "Male"],
            state="readonly",
            style="Modern.TCombobox",
        )
        self.sex_combo.pack(fill="x", pady=(5, 11))
        self.sex_combo.set("Select sex")

        sat_label_row = tk.Frame(subject, bg=c["surface"])
        sat_label_row.pack(fill="x")
        self._field_label(sat_label_row, "Thigh SAT Thickness *").pack(side="left")
        self.sat_help_icon = HelpIcon(
            sat_label_row,
            (
                "EN: SAT = Subcutaneous Adipose Tissue thickness measured at or near "
                "the DCS site. Enter the measured thickness in mm.\n"
                "Current Human V1 accepts values > 0 mm, with Skin + SAT < 40 mm.\n\n"
                "TH: SAT = ความหนาของชั้นเนื้อเยื่อไขมันใต้ผิวหนัง วัดที่หรือใกล้ "
                "ตำแหน่งตรวจ DCS โดยกรอกค่าเป็นมิลลิเมตร\n"
                "Human V1 ปัจจุบันรับค่ามากกว่า 0 mm และ Skin + SAT ต้องน้อยกว่า 40 mm"
            ),
        )
        self.sat_help_icon.pack(side="left", padx=(6, 0))
        sat_entry = ttk.Entry(subject, textvariable=self.sat_var, style="Modern.TEntry")
        sat_entry.pack(fill="x", pady=(5, 2))
        self._helper_label(subject, "mm  •  measured at / near the DCS site").pack(anchor="w", pady=(0, 11))
        self.human_placeholder_entries[sat_entry] = {
            "variable": self.sat_var,
            "placeholder": "e.g. 10.0",
            "active": False,
        }
        self._set_human_placeholder(sat_entry)
        sat_entry.bind("<FocusIn>", lambda e, w=sat_entry: self._clear_human_placeholder(w))
        sat_entry.bind("<FocusOut>", lambda e, w=sat_entry: self._restore_human_placeholder(w))

        skin_label_row = tk.Frame(subject, bg=c["surface"])
        skin_label_row.pack(fill="x")
        self._field_label(
            skin_label_row,
            "Skin Thickness (optional)"
        ).pack(side="left")

        self.skin_help_icon = HelpIcon(
            skin_label_row,
            (
                "EN: Skin Thickness is the measured thickness of the skin layer "
                "at or near the DCS site, entered in mm. This field is optional. "
                "If left blank, Human V1 uses a sex-based reference value.\n"
                "Current Human V1 accepts values > 0 mm, with Skin + SAT < 40 mm.\n\n"
                "TH: Skin Thickness คือความหนาของชั้นผิวหนังที่วัดบริเวณหรือใกล้ "
                "ตำแหน่งตรวจ DCS โดยกรอกค่าเป็นมิลลิเมตร ช่องนี้ไม่จำเป็นต้องกรอก "
                "หากเว้นว่าง Human V1 จะใช้ค่าอ้างอิงตามเพศแทน\n"
                "Human V1 ปัจจุบันรับค่ามากกว่า 0 mm และ Skin + SAT ต้องน้อยกว่า 40 mm"
            ),
        )
        self.skin_help_icon.pack(side="left", padx=(6, 0))

        skin_entry = ttk.Entry(subject, textvariable=self.skin_var, style="Modern.TEntry")
        skin_entry.pack(fill="x", pady=(5, 2))
        self._helper_label(subject, "mm  •  leave blank to use sex-based reference").pack(anchor="w", pady=(0, 13))
        self.human_placeholder_entries[skin_entry] = {
            "variable": self.skin_var,
            "placeholder": "e.g. 1.7",
            "active": False,
        }
        self._set_human_placeholder(skin_entry)
        skin_entry.bind("<FocusIn>", lambda e, w=skin_entry: self._clear_human_placeholder(w))
        skin_entry.bind("<FocusOut>", lambda e, w=skin_entry: self._restore_human_placeholder(w))

        action_row = tk.Frame(subject, bg=c["surface"])
        action_row.pack(fill="x", pady=(2, 0))
        action_row.grid_columnconfigure(0, weight=1, uniform="human_actions")
        action_row.grid_columnconfigure(1, weight=1, uniform="human_actions")

        self.human_generate_button = self._button(
            action_row,
            "Generate Dataset",
            self.run_human_dataset,
            kind="primary",
        )
        self.human_generate_button.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(0, 9),
        )

        self.human_export_csv_button = self._button(
            action_row,
            "Download CSV",
            self.export_human_dataset_csv,
            kind="soft",
            state="disabled",
        )
        self.human_export_csv_button.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=(0, 4),
        )

        self.human_export_mat_button = self._button(
            action_row,
            "Download MAT",
            self.export_human_dataset_mat,
            kind="soft",
            state="disabled",
        )
        self.human_export_mat_button.grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(4, 0),
        )

        self._helper_label(subject, "Downloads unlock when generation is complete.").pack(anchor="w", pady=(8, 0))

        # Generation status intentionally uses ordinary Tk widgets rather than
        # RoundedCard. This isolates the live status from Canvas/card redraws.
        status = tk.Frame(
            left_wrap,
            bg=c["surface"],
            highlightbackground=c["line"],
            highlightthickness=1,
        )
        status.pack(fill="x", pady=(10, 0))

        tk.Label(
            status,
            text="Generation Status",
            bg=c["surface"],
            fg=c["text"],
            font=("Segoe UI Semibold", 13),
        ).pack(anchor="w", padx=18, pady=(11, 0))

        self.human_status_badge = tk.Label(
            status,
            text="Ready",
            bg="#EEF0FF",
            fg=c["violet"],
            font=("Segoe UI Semibold", 9),
            padx=10,
            pady=5,
        )
        self.human_status_badge.pack(anchor="w", padx=18, pady=(8, 8))

        self.human_status_var = tk.StringVar(value="Ready")
        self.human_subject_var = tk.StringVar(value="Subject: -")
        self.human_geometry_var = tk.StringVar(value="Geometry: -")
        self.human_dataset_var = tk.StringVar(value="Dataset: -")

        for var in (self.human_subject_var, self.human_geometry_var, self.human_dataset_var):
            tk.Label(
                status,
                textvariable=var,
                bg=c["surface"],
                fg=c["muted"],
                font=("Segoe UI", 8),
                wraplength=300,
                justify="left",
            ).pack(anchor="w", fill="x", padx=18, pady=(0, 4))

        tk.Frame(status, bg=c["surface"], height=4).pack(fill="x")

        # RIGHT SIDE
        right = tk.Frame(self.human_tab, bg=c["bg"])
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=3)
        right.grid_rowconfigure(1, weight=4)

        # Dataset card
        browser_outer, browser = self._card(right, padx=16, pady=14)
        browser_outer.grid(row=0, column=0, sticky="nsew")
        browser.grid_columnconfigure(0, weight=1)
        browser.grid_rowconfigure(1, weight=1)

        browser_head = tk.Frame(browser, bg=c["surface"])
        browser_head.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Label(
            browser_head,
            text="Dataset Browser",
            bg=c["surface"],
            fg=c["text"],
            font=("Segoe UI Semibold", 14),
        ).pack(anchor="w")
        tk.Label(
            browser_head,
            text="Select a row to preview its shallow / deep flow condition.",
            bg=c["surface"],
            fg=c["muted"],
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(2, 0))

        table_frame = tk.Frame(browser, bg=c["surface"])
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)

        columns = ("condition", "shallow", "deep", "sds", "tau", "g2")
        self.human_dataset_table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
            style="Modern.Treeview",
        )
        headings = {
            "condition": "Condition",
            "shallow": "Shallow Flow",
            "deep": "Deep Flow",
            "sds": "SDS (cm)",
            "tau": "τ (μs)",
            "g2": "g₂",
        }
        widths = {
            "condition": 90,
            "shallow": 115,
            "deep": 115,
            "sds": 85,
            "tau": 90,
            "g2": 130,
        }
        for column in columns:
            self.human_dataset_table.heading(column, text=headings[column])
            self.human_dataset_table.column(
                column,
                width=widths[column],
                minwidth=70,
                anchor="center",
                stretch=True,
            )

        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.human_dataset_table.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.human_dataset_table.xview)
        self.human_dataset_table.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.human_dataset_table.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.human_dataset_table.bind("<<TreeviewSelect>>", self.on_human_dataset_row_selected)

        # Graph card
        graph_outer, graph = self._card(right, padx=16, pady=14)
        graph_outer.grid(row=1, column=0, sticky="nsew", pady=(14, 0))
        graph.grid_columnconfigure(0, weight=1)
        graph.grid_rowconfigure(1, weight=1)

        graph_head = tk.Frame(graph, bg=c["surface"])
        graph_head.grid(row=0, column=0, sticky="ew", pady=(0, 7))
        graph_head.grid_columnconfigure(0, weight=1)

        tk.Label(
            graph_head,
            text="g₂ Graph",
            bg=c["surface"],
            fg=c["text"],
            font=("Segoe UI Semibold", 14),
        ).grid(row=0, column=0, sticky="w")

        self.human_preview_shallow_var = tk.StringVar()
        self.human_preview_deep_var = tk.StringVar()

        tk.Label(
            graph_head,
            text="Shallow Flow Rate\nอัตราการไหลชั้นตื้น",
            bg=c["surface"],
            fg=c["muted"],
            font=("TH Sarabun New", 13),
        ).grid(row=0, column=1, padx=(8, 4))

        self.human_preview_shallow_combo = ttk.Combobox(
            graph_head,
            textvariable=self.human_preview_shallow_var,
            state="readonly",
            width=7,
            style="Modern.TCombobox",
        )
        self.human_preview_shallow_combo.grid(row=0, column=2)

        tk.Label(
            graph_head,
            text="Deep Flow Rate\nอัตราการไหลชั้นลึก",
            bg=c["surface"],
            fg=c["muted"],
            font=("TH Sarabun New", 13),
        ).grid(row=0, column=3, padx=(12, 4))

        self.human_preview_deep_combo = ttk.Combobox(
            graph_head,
            textvariable=self.human_preview_deep_var,
            state="readonly",
            width=7,
            style="Modern.TCombobox",
        )
        self.human_preview_deep_combo.grid(row=0, column=4)

        tk.Label(
            graph_head,
            text="ml/min/100g",
            bg=c["surface"],
            fg=c["muted"],
            font=("Segoe UI", 8),
        ).grid(row=0, column=5, padx=(7, 0))

        self.human_preview_shallow_combo.bind("<<ComboboxSelected>>", self.update_human_preview)
        self.human_preview_deep_combo.bind("<<ComboboxSelected>>", self.update_human_preview)

        graph_frame = tk.Frame(graph, bg=c["surface"])
        graph_frame.grid(row=1, column=0, sticky="nsew")
        graph_frame.grid_columnconfigure(0, weight=1)
        graph_frame.grid_rowconfigure(0, weight=1)

        self.human_figure = Figure(figsize=(9.2, 3.7), dpi=100, facecolor=c["surface"])
        self.human_axis = self.human_figure.add_subplot(111)
        self._style_human_axis("Generate a dataset to preview")

        self.human_canvas = FigureCanvasTkAgg(self.human_figure, master=graph_frame)
        self.human_canvas.draw()
        self.human_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _style_human_axis(self, title=None):
        c = self.COLORS
        ax = self.human_axis
        ax.set_facecolor("#FBFCFF")
        ax.set_xlabel("Delay Time τ (μs)", color=c["muted"])
        ax.set_ylabel("g₂(τ)", color=c["muted"])
        if title is not None:
            ax.set_title(title, color=c["text"], fontsize=12, pad=12)
        ax.grid(True, color="#E8EAF2", linewidth=0.8)
        ax.tick_params(colors=c["muted"], labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#E0E3EC")

    def _set_human_placeholder(self, entry):
        info = self.human_placeholder_entries[entry]
        info["active"] = True
        entry.configure(foreground="#888888")
        info["variable"].set(info["placeholder"])


    def _clear_human_placeholder(self, entry):
        info = self.human_placeholder_entries[entry]

        if info["active"]:
            info["variable"].set("")
            entry.configure(foreground="#000000")
            info["active"] = False


    def _restore_human_placeholder(self, entry):
        info = self.human_placeholder_entries[entry]

        if not info["variable"].get().strip():
            self._set_human_placeholder(entry)


    def _human_entry_value(self, variable):
        for entry, info in self.human_placeholder_entries.items():
            if info["variable"] is variable:
                if info["active"]:
                    return ""
                return variable.get().strip()

        return variable.get().strip()


    def run_human_dataset(self):
        try:
            sex = self.sex_var.get().strip()
            sat_text = self._human_entry_value(self.sat_var)
            skin_text = self._human_entry_value(self.skin_var)

            missing = []

            if sex not in ("Female", "Male"):
                missing.append("Sex")

            if not sat_text:
                missing.append("Thigh SAT Thickness")

            if missing:
                raise ValueError(
                    "Please complete: " + ", ".join(missing) + "."
                )

            try:
                sat = float(sat_text)
            except ValueError:
                raise ValueError(
                    "Thigh SAT Thickness must be a number, "
                    "for example 10.0 mm."
                )

            if not np.isfinite(sat):
                raise ValueError(
                    "Thigh SAT Thickness must be a finite number."
                )

            if sat <= 0:
                raise ValueError(
                    "Thigh SAT Thickness must be greater than 0 mm."
                )

            skin = None

            if skin_text:
                try:
                    skin = float(skin_text)
                except ValueError:
                    raise ValueError(
                        "Skin Thickness must be a number, "
                        "for example 1.7 mm."
                    )

                if not np.isfinite(skin):
                    raise ValueError(
                        "Skin Thickness must be a finite number."
                    )

                if skin <= 0:
                    raise ValueError(
                        "Skin Thickness must be greater than 0 mm."
                    )

            fallback_skin = 1.89 if sex == "Male" else 1.65
            effective_skin = skin if skin is not None else fallback_skin
            superficial_depth = sat + effective_skin

            if superficial_depth >= 40.0:
                raise ValueError(
                    "Skin + SAT is too thick for the current Human V1 "
                    "simulation domain. Please use measurements whose "
                    "combined superficial depth is below 40 mm."
                )

            shallow = [5.0, 10.0, 20.0]
            deep = [5.0, 10.0, 20.0]

            human = HumanInput(
                sex=sex,
                sat_thickness_mm=sat,
                skin_thickness_mm=skin,
            )


            self.human_generate_button.config(state="disabled")
            self.human_export_csv_button.config(state="disabled")
            self.human_export_mat_button.config(state="disabled")

            self.human_status_var.set("Running MCX and generating dataset...")
            self._set_status_badge(
                self.human_status_badge,
                "Generating...",
                "processing",
            )

            # Same repaint + background-worker pattern that passed the
            # standalone real generate_human_dataset()/MCX test.
            self.root.update()

            threading.Thread(
                target=self._human_dataset_worker,
                args=(human, shallow, deep),
                daemon=True,
            ).start()

        except Exception as error:
            self._human_dataset_failed(error)



    def _human_dataset_worker(self, human, shallow, deep):
        try:
            output = generate_human_dataset(
                human=human, shallow_flows=shallow, deep_flows=deep,
                nphoton=10_000_000, export=False)
            self.root.after(0, lambda: self._human_dataset_complete(output))
        except Exception as error:
            self.root.after(
                0, lambda err=error: self._human_dataset_failed(err))


    def _human_dataset_complete(self, output):
        self.last_human_output = output
        self.human_dataset = output["dataset"]

        dataset = self.human_dataset
        geometry = output["geometry"]
        layer_info = output["layer_info"]

        shallow_values = sorted(
            dataset["flow_shallow"].unique()
        )
        deep_values = sorted(
            dataset["flow_deep"].unique()
        )

        shallow_display = [
            format_flow_value(value)
            for value in shallow_values
        ]
        deep_display = [
            format_flow_value(value)
            for value in deep_values
        ]

        self.human_preview_shallow_combo["values"] = shallow_display
        self.human_preview_deep_combo["values"] = deep_display

        self.human_preview_shallow_var.set(shallow_display[0])
        self.human_preview_deep_var.set(deep_display[0])

        conditions = len(shallow_values) * len(deep_values)
        curves = conditions * len(
            sorted(dataset["sds_cm"].unique())
        )

        self.human_subject_var.set(
            f"Subject: {geometry.sex}  |  "
            f"SAT {geometry.sat_thickness_mm:.2f} mm"
        )

        self.human_geometry_var.set(
            f"Skin: {geometry.skin_thickness_mm:.2f} mm "
            f"({geometry.skin_thickness_source})  |  "
            f"Depth: {geometry.superficial_thickness_mm:.2f} mm "
            f"→ voxelized {layer_info['shallow_depth_mm']:.2f} mm"
        )

        self.human_dataset_var.set(
            f"Dataset: {conditions} conditions, "
            f"{curves} curves, "
            f"{len(dataset):,} rows"
        )

        self.human_status_var.set("Complete")
        self._set_status_badge(self.human_status_badge, "Complete", "success")

        self.human_generate_button.config(state="normal")
        self.human_export_csv_button.config(state="normal")
        self.human_export_mat_button.config(state="normal")

        self.populate_human_dataset_table()
        self.update_human_preview()


    def _human_dataset_failed(self, error):
        self.human_status_var.set("Error")
        self._set_status_badge(self.human_status_badge, "Failed", "error")
        self.human_generate_button.config(state="normal")
        self.human_export_csv_button.config(state="disabled")
        self.human_export_mat_button.config(state="disabled")
        messagebox.showerror(
            "Human Dataset Error",
            str(error),
        )


    def populate_human_dataset_table(self):
        if self.human_dataset is None:
            return

        for item in self.human_dataset_table.get_children():
            self.human_dataset_table.delete(item)

        dataset = self.human_dataset.sort_values(
            [
                "flow_shallow",
                "flow_deep",
                "sds_cm",
                "tau_us",
            ]
        )

        condition_map = {}
        condition_number = 0

        for _, row in dataset.iterrows():
            key = (
                float(row["flow_shallow"]),
                float(row["flow_deep"]),
            )

            if key not in condition_map:
                condition_number += 1
                condition_map[key] = condition_number

            self.human_dataset_table.insert(
                "",
                "end",
                values=(
                    condition_map[key],
                    format_flow_value(
                        row["flow_shallow"]
                    ),
                    format_flow_value(
                        row["flow_deep"]
                    ),
                    f"{row['sds_cm']:.0f}",
                    f"{row['tau_us']:.0f}",
                    f"{row['g2']:.8f}",
                ),
            )


    def on_human_dataset_row_selected(self, event=None):
        selection = self.human_dataset_table.selection()

        if not selection:
            return

        values = self.human_dataset_table.item(
            selection[0],
            "values",
        )

        if not values:
            return

        self.human_preview_shallow_var.set(
            str(values[1])
        )
        self.human_preview_deep_var.set(
            str(values[2])
        )

        self.update_human_preview()


    def update_human_preview(self, event=None):
        if self.human_dataset is None:
            return

        try:
            shallow = float(
                self.human_preview_shallow_var.get()
            )
            deep = float(
                self.human_preview_deep_var.get()
            )

            dataset = self.human_dataset

            condition = dataset.loc[
                np.isclose(
                    dataset["flow_shallow"],
                    shallow,
                )
                & np.isclose(
                    dataset["flow_deep"],
                    deep,
                )
            ]

            if condition.empty:
                raise ValueError(
                    "Selected flow condition was not found."
                )

            self.human_axis.clear()
            self.human_axis.set_facecolor("#FBFCFF")

            plot_colors = ["#6D5DFB", "#16B8A6", "#F59E0B"]
            for color_index, sds in enumerate(sorted(
                condition["sds_cm"].unique()
            )):
                curve = condition.loc[
                    condition["sds_cm"] == sds
                ].sort_values("tau_us")

                self.human_axis.plot(
                    curve["tau_us"].to_numpy(),
                    curve["g2"].to_numpy(),
                    linewidth=2.3,
                    color=plot_colors[color_index % len(plot_colors)],
                    label=f"SDS {sds:g} cm",
                )

            self.human_axis.set_title(
                f"Human Synthetic g₂  |  "
                f"Shallow Flow = {shallow:g}, "
                f"Deep Flow = {deep:g}",
                fontsize=12,
                pad=12,
            )

            self.human_axis.set_xlabel(
                "Delay Time τ (μs)"
            )
            self.human_axis.set_ylabel(
                "g₂(τ)"
            )

            self.human_axis.grid(True, color="#E8EAF2", linewidth=0.8)
            self.human_axis.tick_params(colors=self.COLORS["muted"], labelsize=9)
            for spine in self.human_axis.spines.values():
                spine.set_color("#E0E3EC")

            self.human_axis.legend(
                loc="upper right",
                ncol=3,
                fontsize=9,
                frameon=False,
            )

            tau_min = float(
                condition["tau_us"].min()
            )
            tau_max = float(
                condition["tau_us"].max()
            )

            self.human_axis.set_xlim(
                tau_min,
                tau_max,
            )

            self.human_axis.margins(x=0)

            self.human_figure.subplots_adjust(
                left=0.085,
                right=0.985,
                bottom=0.18,
                top=0.88,
            )

            self.human_canvas.draw()

        except Exception as error:
            messagebox.showerror(
                "Preview Error",
                str(error),
            )


    def export_human_dataset_csv(self):
        if self.human_dataset is None:
            messagebox.showinfo(
                "Download Dataset",
                "Generate a dataset first.",
            )
            return

        filename = filedialog.asksaveasfilename(
            title="Download Human DCS Dataset as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="human_dcs_dataset.csv",
        )

        if not filename:
            return

        try:
            self.human_dataset.to_csv(filename, index=False)
            messagebox.showinfo(
                "Download Complete",
                f"CSV dataset saved to:\n{filename}",
            )
        except Exception as error:
            messagebox.showerror("CSV Export Error", str(error))


    def export_human_dataset_mat(self):
        if self.human_dataset is None or self.last_human_output is None:
            messagebox.showinfo(
                "Download Dataset",
                "Generate a dataset first.",
            )
            return

        filename = filedialog.asksaveasfilename(
            title="Download Human DCS Dataset as MAT",
            defaultextension=".mat",
            filetypes=[("MATLAB MAT files", "*.mat")],
            initialfile="human_dcs_dataset.mat",
        )

        if not filename:
            return

        try:
            dataset = self.human_dataset
            geometry = self.last_human_output["geometry"]
            layer_info = self.last_human_output["layer_info"]

            # Store both the numerical dataset and useful metadata.
            # Strings are stored separately because MATLAB handles
            # mixed-type table-like data differently from pandas.
            mat_data = {
                "sex": str(geometry.sex),
                "skin_thickness_mm": float(geometry.skin_thickness_mm),
                "skin_thickness_source": str(geometry.skin_thickness_source),
                "sat_thickness_mm": float(geometry.sat_thickness_mm),
                "requested_superficial_depth_mm": float(
                    geometry.superficial_thickness_mm
                ),
                "voxelized_superficial_depth_mm": float(
                    layer_info["shallow_depth_mm"]
                ),
                "deep_region_depth_mm": float(
                    layer_info["deep_depth_mm"]
                ),
                "flow_shallow": dataset["flow_shallow"].to_numpy(dtype=float),
                "flow_deep": dataset["flow_deep"].to_numpy(dtype=float),
                "alpha_db_shallow_mm2_s": dataset[
                    "alpha_db_shallow_mm2_s"
                ].to_numpy(dtype=float),
                "alpha_db_deep_mm2_s": dataset[
                    "alpha_db_deep_mm2_s"
                ].to_numpy(dtype=float),
                "sds_cm": dataset["sds_cm"].to_numpy(dtype=float),
                "tau_us": dataset["tau_us"].to_numpy(dtype=float),
                "g2": dataset["g2"].to_numpy(dtype=float),
                "column_names": np.array(
                    list(dataset.columns),
                    dtype=object,
                ),
                "generator_version": "Human DCS Simulator V1",
                "notes": (
                    "Synthetic forward-model dataset. "
                    "Flow values are assumed simulation conditions; "
                    "this file does not estimate blood flow from measured g2."
                ),
            }

            savemat(filename, mat_data)

            messagebox.showinfo(
                "Download Complete",
                f"MAT dataset saved to:\n{filename}",
            )

        except Exception as error:
            messagebox.showerror("MAT Export Error", str(error))

# ============================================================
# MAIN
# ============================================================

def main():

    root = tk.Tk()

    DCSSimulatorGUI(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()