from pathlib import Path

import ale_py
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


class A2CService:
    def train(self, form, callback):
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

        policy_kwargs = {}
        if use_rms_prop:
            policy_kwargs = {
                "optimizer_class": RMSpropTFLike,
                "optimizer_kwargs": {"eps": rms_prop_eps},
            }

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
                learning_rate=learning_rate,
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                rms_prop_eps=rms_prop_eps,
                use_rms_prop=use_rms_prop,
                policy_kwargs=policy_kwargs,
                normalize_advantage=normalize_advantage,
                seed=seed,
                verbose=1,
                tensorboard_log=str(Path(__file__).resolve().parent.parent / "trained_models_logs"),
            )

            model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                tb_log_name=f"a2c_{form['model_name']}",
            )
            return model
        finally:
            env.close()
