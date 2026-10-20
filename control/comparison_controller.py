from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class ComparisonController:
    def __init__(self):
        self.logs_dir = Path(__file__).resolve().parent.parent / "trained_models_logs"

    def get_runs(self):
        return sorted(
            path.name for path in self.logs_dir.glob("*")
            if path.is_dir() and any(path.glob("events.out.tfevents.*"))
        )

    def compare(self, run_names):
        plt.rcParams["savefig.dpi"] = 300
        plt.figure(figsize=(9, 5))
        for name in run_names:
            events = EventAccumulator(
                str(self.logs_dir / name), size_guidance={"scalars": 0}
            )
            events.Reload()
            if "rollout/ep_rew_mean" not in events.Tags()["scalars"]:
                continue
            points = events.Scalars("rollout/ep_rew_mean")
            steps = [point.step for point in points]
            rewards = [point.value for point in points]
            plt.plot(steps, rewards, label=name)
        plt.xlabel("Liczba kroków treningu")
        plt.ylabel("Średni wynik epizodu")
        plt.title("Porównanie przebiegu uczenia — Space Invaders")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=False)
