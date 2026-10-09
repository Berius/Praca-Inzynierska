import customtkinter as ctk


class TrainMenu(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

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
            values=["PPO", "A2C", "DQN"]
        )

        self.algorithm_select.pack(pady=10)
        self.algorithm_select.set("PPO")

        form_frame = ctk.CTkFrame(self)
        form_frame.pack(pady=20, padx=20)

        fields = [
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

        self.entries = {}

        for row, (name, label_text, default_value) in enumerate(fields):
            label = ctk.CTkLabel(
                form_frame,
                text=label_text
            )

            entry = ctk.CTkEntry(
                form_frame
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
