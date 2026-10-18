import ale_py
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


def linear_schedule(initial_value):
    return lambda progress_remaining: max(0.0, progress_remaining) * initial_value


class A2CService:
    def train(self, form):
        total_timesteps = int(form["total_timesteps"])
        learning_rate = float(form["learning_rate"])
        n_steps = int(form["n_steps"])
        gamma = float(form["gamma"])
        gae_lambda = float(form["gae_lambda"])
        ent_coef = float(form["ent_coef"])
        vf_coef = float(form["vf_coef"])
        max_grad_norm = float(form["max_grad_norm"])
        rms_prop_eps = float(form["rms_prop_eps"])
        use_rms_prop = str(form["use_rms_prop"]).lower() in ("true", "1")
        normalize_advantage = str(form["normalize_advantage"]).lower() in ("true", "1")
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
            model = A2C(
                "CnnPolicy",
                env,
                learning_rate=linear_schedule(learning_rate),
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                rms_prop_eps=rms_prop_eps,
                use_rms_prop=use_rms_prop,
                normalize_advantage=normalize_advantage,
                seed=seed,
                verbose=1,
                tensorboard_log="./trained_models_logs/",
            )

            model.learn(total_timesteps=total_timesteps)
            model.save("trained_models/a2c")
        finally:
            env.close()
