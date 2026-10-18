from threading import Thread

from algorithms.ppo import PPOService
from algorithms.dqn import DQNService
from algorithms.a2c import A2CService


class TrainController:
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

        Thread(target=service.train, args=(form,), daemon=True).start()
