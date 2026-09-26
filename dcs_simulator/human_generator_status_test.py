import tkinter as tk
import threading
import traceback

# This test intentionally imports the REAL project generator.
from dataset_generator import generate_human_dataset
from human_model import HumanInput


class HumanGeneratorStatusTest:
    def __init__(self, root):
        self.root = root
        root.title("Human Generator Status Test")
        root.geometry("560x360")
        root.configure(bg="#F6F7FB")

        card = tk.Frame(root, bg="white", highlightthickness=1,
                        highlightbackground="#E5E7EB")
        card.pack(fill="both", expand=True, padx=28, pady=28)

        tk.Label(
            card, text="Real Human Generator Test",
            bg="white", fg="#172033",
            font=("Segoe UI Semibold", 18)
        ).pack(anchor="w", padx=24, pady=(24, 4))

        tk.Label(
            card,
            text="Calls the real generate_human_dataset() with 10,000,000 photons.",
            bg="white", fg="#667085",
            font=("Segoe UI", 9)
        ).pack(anchor="w", padx=24)

        self.status = tk.Label(
            card, text="Ready",
            bg="#EEF0FF", fg="#6D5DFB",
            font=("Segoe UI Semibold", 10),
            padx=14, pady=7
        )
        self.status.pack(anchor="w", padx=24, pady=(22, 12))

        self.detail = tk.StringVar(value="Press the button to run the real generator.")
        tk.Label(
            card, textvariable=self.detail,
            bg="white", fg="#667085",
            font=("Segoe UI", 9),
            wraplength=480, justify="left"
        ).pack(anchor="w", padx=24, pady=(0, 18))

        self.button = tk.Button(
            card,
            text="Run Real Human Generator",
            command=self.start,
            bg="#6D5DFB", fg="white",
            activebackground="#5949E8",
            activeforeground="white",
            relief="flat", bd=0,
            font=("Segoe UI Semibold", 10),
            padx=18, pady=10,
            cursor="hand2"
        )
        self.button.pack(fill="x", padx=24, pady=(0, 24))

    def set_status(self, text, bg, fg):
        self.status.configure(text=text, bg=bg, fg=fg)

    def start(self):
        self.button.config(state="disabled")
        self.set_status("Generating...", "#FFF7E8", "#F59E0B")
        self.detail.set("MCX is running. Watch whether this status remains visible.")
        print("GUI STATUS BEFORE WORKER:", self.status.cget("text"))

        # Force Tk to paint before launching the real MCX work.
        self.root.update()

        human = HumanInput(
            sex="Female",
            sat_thickness_mm=10.0,
        )

        threading.Thread(
            target=self.worker,
            args=(human,),
            daemon=True
        ).start()

    def worker(self, human):
        try:
            result = generate_human_dataset(
                human=human,
                shallow_flows=[5.0, 10.0, 20.0],
                deep_flows=[5.0, 10.0, 20.0],
                nphoton=10_000_000,
                export=False,
            )
            self.root.after(0, lambda: self.complete(result))
        except Exception as exc:
            traceback.print_exc()
            self.root.after(0, lambda e=exc: self.failed(e))

    def complete(self, result):
        rows = len(result["dataset"])
        self.set_status("Complete", "#E8FBF7", "#10B981")
        self.detail.set(f"Generation completed successfully. Rows: {rows:,}")
        self.button.config(state="normal")
        print("GUI STATUS AFTER WORKER: Complete")

    def failed(self, exc):
        self.set_status("Failed", "#FFF0F0", "#EF4444")
        self.detail.set(f"{type(exc).__name__}: {exc}")
        self.button.config(state="normal")
        print("GUI STATUS AFTER WORKER: Failed")


def main():
    root = tk.Tk()
    HumanGeneratorStatusTest(root)
    root.mainloop()


if __name__ == "__main__":
    main()
