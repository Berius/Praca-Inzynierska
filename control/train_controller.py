from pathlib import Path
from queue import Queue
from threading import Event, Thread

from stable_baselines3.common.callbacks import BaseCallback

from algorithms.ppo import PPOService
from algorithms.dqn import DQNService
from algorithms.a2c import A2CService


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, progress_queue, stop_event):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.progress_queue = progress_queue
        self.stop_event = stop_event
        self.last_update = 0

    def _on_step(self):
        current = self.num_timesteps
        if current - self.last_update >= 1000 or current >= self.total_timesteps:
            self.progress_queue.put(("progress", current))
            self.last_update = current
        return not self.stop_event.is_set()


class TrainController:
    def __init__(self, on_training_start):
        self.on_training_start = on_training_start

    def start_training(self, form):
        algorithm = form["algorithm"]

        if algorithm == "PPO":
            service = PPOService()
        elif algorithm == "DQN":
            service = DQNService()
        elif algorithm == "A2C":
            service = A2CService()
        else:
            print(f"Nieznany algorytm: {algorithm}")
            return

        progress_queue = Queue()
        stop_event = Event()
        callback = ProgressCallback(int(form["total_timesteps"]), progress_queue, stop_event)
        self.on_training_start(form, progress_queue, stop_event)
        Thread(target=self.train_model, args=(service, form, callback), daemon=True).start()

    def train_model(self, service, form, callback):
        model = service.train(form, callback)
        if callback.stop_event.is_set():
            callback.progress_queue.put(("stopped", None))
            return
        self.save_model(model, form)
        callback.progress_queue.put(("done", None))

    def save_model(self, model, form):
        models_dir = Path(__file__).resolve().parent.parent / "trained_models"
        models_dir.mkdir(exist_ok=True)
        model_name = f"{form['algorithm'].lower()}_{form['model_name']}"
        model.save(models_dir / model_name)
