from pathlib import Path

import ale_py
import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage


class Play:
    def play(self, model_path):
        model_path = Path(model_path)
        name = model_path.stem.lower()
        if name.startswith("ppo"):
            algorithm = PPO
        elif name.startswith("dqn"):
            algorithm = DQN
        elif name.startswith("a2c"):
            algorithm = A2C
        else:
            print("Nazwa modelu powinna zaczynać się od ppo, dqn lub a2c.")
            return

        gym.register_envs(ale_py)
        env = make_atari_env(
            "ALE/SpaceInvaders-v5",
            n_envs=1,
            env_kwargs={
                "render_mode": "human",
                "frameskip": 1,
                "repeat_action_probability": 0.25,
                "full_action_space": False,
            },
            wrapper_kwargs={
                "frame_skip": 4,
                "screen_size": 84,
                "clip_reward": False,
                "terminal_on_life_loss": False,
            },
        )

        try:
            env = VecFrameStack(env, n_stack=4)
            env = VecTransposeImage(env)
            model = algorithm.load(str(model_path), env=env)
            observation = env.reset()
            score = 0.0
            done = False

            while not done:
                action, _ = model.predict(observation, deterministic=False)
                observation, rewards, dones, _ = env.step(action)
                score += float(rewards[0])
                done = bool(dones[0])

            print(f"Model {model_path.name}: wynik = {score:.0f}")
        finally:
            env.close()
