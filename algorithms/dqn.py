from pathlib import Path

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


class DQNService:
    def train(self, form, callback):
        total_timesteps = int(form["total_timesteps"])
        learning_rate = float(form["learning_rate"])
        buffer_size = int(form["buffer_size"])
        learning_starts = int(form["learning_starts"])
        tau = float(form["tau"])
        train_freq = int(form["train_freq"])
        gradient_steps = int(form["gradient_steps"])
        target_update_interval = int(form["target_update_interval"])
        exploration_fraction = float(form["exploration_fraction"])
        exploration_initial_eps = float(form["exploration_initial_eps"])
        exploration_final_eps = float(form["exploration_final_eps"])
        max_grad_norm = float(form["max_grad_norm"])
        batch_size = int(form["batch_size"])
        gamma = float(form["gamma"])
        seed = int(form["seed"])

        gym.register_envs(ale_py)

        env = make_atari_env(
            "ALE/SpaceInvaders-v5",
            n_envs=8,
            seed=seed,
            env_kwargs={
                "frameskip": 1,
                "repeat_action_probability": 0.25,
                "full_action_space": False,
            },
            wrapper_kwargs={
                "frame_skip": 4,
                "screen_size": 84,
                "clip_reward": True,
                "terminal_on_life_loss": True,
            },
        )
        env = VecFrameStack(env, n_stack=4)

        try:
            model = DQN(
                "CnnPolicy",
                env,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                tau=tau,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                target_update_interval=target_update_interval,
                exploration_fraction=exploration_fraction,
                exploration_initial_eps=exploration_initial_eps,
                exploration_final_eps=exploration_final_eps,
                max_grad_norm=max_grad_norm,
                batch_size=batch_size,
                gamma=gamma,
                seed=seed,
                verbose=1,
                tensorboard_log=str(Path(__file__).resolve().parent.parent / "trained_models_logs"),
            )

            model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                tb_log_name=f"dqn_{form['model_name']}",
            )
            return model
        finally:
            env.close()
