import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


def linear_schedule(initial_value):
    return lambda progress_remaining: progress_remaining * initial_value


class PPOService:
    def train(self, form):
        total_timesteps = int(form["total_timesteps"])
        learning_rate = float(form["learning_rate"])
        n_steps = int(form["n_steps"])
        batch_size = int(form["batch_size"])
        n_epochs = int(form["n_epochs"])
        gamma = float(form["gamma"])
        gae_lambda = float(form["gae_lambda"])
        clip_range = float(form["clip_range"])
        ent_coef = float(form["ent_coef"])
        vf_coef = float(form["vf_coef"])
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

        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=linear_schedule(learning_rate),
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=linear_schedule(clip_range),
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            seed=seed,
            verbose=1,
        )

        model.learn(total_timesteps=total_timesteps)
        model.save("trained_models/ppo_space_invaders")
        env.close()
