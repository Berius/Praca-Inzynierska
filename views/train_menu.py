import customtkinter as ctk


class TrainMenu(ctk.CTkFrame):
    def __init__(self, parent, on_back_click, on_train_click):
        super().__init__(parent, fg_color="black", corner_radius=0)

        font_arcade = ctk.CTkFont(
            family="Consolas",
            size=24,
            weight="bold"
        )

        title = ctk.CTkLabel(
            self,
            text="Trenuj",
            font=font_arcade
        )

        title.pack(pady=(20, 10))

        self.algorithm_select = ctk.CTkComboBox(
            self,
            values=["PPO", "A2C", "DQN"],
            command=self.update_fields,
        )

        self.algorithm_select.pack(pady=10)
        self.algorithm_select.set("PPO")

        self.form_frame = ctk.CTkFrame(self)
        self.form_frame.pack(pady=10, padx=20)

        self.train_button = ctk.CTkButton(
            self,
            text="Trenuj",
            command=lambda: on_train_click(self.get_form())
        )

        self.train_button.pack(pady=(0, 10))

        self.back_button = ctk.CTkButton(
            self,
            text="Powrót",
            command=on_back_click
        )

        self.back_button.pack(pady=(0, 20))
        self.update_fields("PPO")

    def update_fields(self, algorithm):
        fields_ppo = [
            ("total_timesteps", "Total timesteps", "1000000"),
            ("learning_rate", "Learning rate", "0.0003"),
            ("n_steps", "n_steps", "2048"),
            ("batch_size", "Batch size", "64"),
            ("n_epochs", "n_epochs", "10"),
            ("gamma", "Gamma", "0.99"),
            ("gae_lambda", "GAE lambda", "0.95"),
            ("clip_range", "Clip range", "0.2"),
            ("ent_coef", "Ent coef", "0.0"),
            ("vf_coef", "VF coef", "0.5"),
            ("seed", "Seed", "42")
        ]

        fields_dqn = [
            ("total_timesteps", "Total timesteps", "1000000"),
            ("learning_rate", "Learning rate", "0.0001"),
            ("batch_size", "Batch size", "32"),
            ("gamma", "Gamma", "0.99"),
            ("seed", "Seed", "42"),
            ("buffer_size", "Buffer size", "10000"),
            ("learning_starts", "Learning starts", "1000"),
            ("tau", "Tau", "1.0"),
            ("train_freq", "Train frequency", "4"),
            ("gradient_steps", "Gradient steps", "1"),
            ("target_update_interval", "Target update interval", "10000"),
            ("exploration_fraction", "Exploration fraction", "0.1"),
            ("exploration_initial_eps", "Exploration initial epsilon", "1.0"),
            ("exploration_final_eps", "Exploration final epsilon", "0.05"),
            ("max_grad_norm", "Max gradient norm", "10.0")
        ]

        fields_a2c = [
            ("total_timesteps", "Total timesteps", "1000000"),
            ("learning_rate", "Learning rate", "0.0007"),
            ("n_steps", "n_steps", "5"),
            ("gamma", "Gamma", "0.99"),
            ("gae_lambda", "GAE lambda", "1.0"),
            ("ent_coef", "Ent coef", "0.0"),
            ("vf_coef", "VF coef", "0.5"),
            ("max_grad_norm", "Max gradient norm", "0.5"),
            ("rms_prop_eps", "RMSprop epsilon", "0.00001"),
            ("use_rms_prop", "Use RMSprop (True/False)", "True"),
            ("normalize_advantage", "Normalize advantage (True/False)", "False"),
            ("seed", "Seed", "42")
        ]

        if algorithm == "PPO":
            fields = fields_ppo
        elif algorithm == "DQN":
            fields = fields_dqn
        elif algorithm == "A2C":
            fields = fields_a2c
        else:
            fields = []

        for widget in self.form_frame.winfo_children():
            widget.destroy()
        self.entries = {}

        for row, (name, label_text, default_value) in enumerate(fields):
            label = ctk.CTkLabel(
                self.form_frame,
                text=label_text
            )

            entry = ctk.CTkEntry(
                self.form_frame
            )

            entry.insert(0, default_value)

            label.grid(
                row=row,
                column=0,
                padx=10,
                pady=5
            )

            entry.grid(
                row=row,
                column=1,
                padx=10,
                pady=5
            )

            self.entries[name] = entry

    def get_form(self):
        form = {name: entry.get() for name, entry in self.entries.items()}
        form["algorithm"] = self.algorithm_select.get()
        return form
