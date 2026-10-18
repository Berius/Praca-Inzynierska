from pathlib import Path
from threading import Thread

from algorithms.play import Play


class ModelSelectionController:
    def __init__(self):
        self.models_dir = Path(__file__).resolve().parent.parent / "trained_models"

    def get_models(self):
        return sorted(path.name for path in self.models_dir.glob("*.zip") if path.is_file())

    def play(self, model_name):
        model_path = self.models_dir / model_name
        Thread(target=Play().play, args=(model_path,), daemon=True).start()
