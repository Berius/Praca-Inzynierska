import customtkinter as ctk
from CTkToolTip import CTkToolTip

PPO_TOOLTIPS = {
    "total_timesteps": (
        "Określa, ile łącznie kroków ma wykonać agent podczas treningu. "
        "Im większa wartość, tym dłużej model będzie się uczył. "
        "Liczba kroków jest sumowana ze wszystkich 8 środowisk."
    ),

    "learning_rate": (
        "Określa, jak mocno model zmienia swoje wagi podczas uczenia. "
        "Większa wartość oznacza większe zmiany, ale może powodować niestabilność. "
        "Mniejsza wartość oznacza ostrożniejsze uczenie. "
    ),

    "n_steps": (
        "Określa, ile kroków wykona każde środowisko, zanim model rozpocznie aktualizację. "
        "Większa wartość oznacza więcej danych do jednej fazy uczenia, "
        "ale także większe zużycie pamięci."
    ),

    "batch_size": (
        "Określa, ile próbek model analizuje jednocześnie podczas jednej aktualizacji wag. "
        "Większa wartość zwykle daje bardziej stabilne aktualizacje, "
        "Mniejsza wartość oznacza bardziej zmienne aktualizacje."
    ),

    "n_epochs": (
        "Określa, ile razy model uczy się na tym samym zestawie zebranych danych. "
        "Większa wartość pozwala intensywniej wykorzystać próbki, "
        "ale wydłuża uczenie i może prowadzić do nadmiernego dopasowania."
    ),

    "gamma": (
        "Określa, jak ważne są przyszłe nagrody dla agenta. "
        "Wartość bliska 1 sprawia, że agent mocniej uwzględnia odległe nagrody. "
        "Mniejsza wartość powoduje skupienie na nagrodach otrzymywanych szybciej."
    ),

    "gae_lambda": (
        "Określa, jak model wykorzystuje nagrody z kolejnych kroków "
        "do oceny, czy wykonana akcja była korzystna. "
        "Wartość bliższa 1 uwzględnia więcej informacji z dalszych kroków, "
        "ale może zwiększać zmienność ocen. "
        "Mniejsza wartość opiera ocenę bardziej na krótkoterminowych przewidywaniach."
    ),

    "clip_range": (
        "Określa, jak mocno PPO ogranicza zmiany sposobu wybierania akcji "
        "podczas aktualizacji modelu. "
        "Mniejsza wartość wymusza ostrożniejsze zmiany. "
        "Większa pozwala na większe zmiany, ale może pogorszyć stabilność uczenia. "
    ),

    "ent_coef": (
        "Określa, jak mocno model jest zachęcany do próbowania różnych akcji. "
        "Większa wartość sprzyja eksploracji zamiast wybierania ciągle tych samych akcji. "
        "Wartość 0 wyłącza dodatkową zachętę do eksploracji."
    ),

    "vf_coef": (
        "Określa, jak duże znaczenie podczas uczenia ma poprawność "
        "przewidywania przyszłych nagród. "
        "Większa wartość zwiększa nacisk na uczenie tych przewidywań. "
        "Mniejsza zmniejsza ich wpływ na całkowitą funkcję straty."
    ),

    "seed": (
        "Ustawia ziarno losowości używane podczas treningu. "
        "Ta sama wartość pomaga porównywać eksperymenty w podobnych warunkach. "
        "Zmiana wartości może prowadzić do innych wyników treningu. "
    ),
}


class TrainMenu(ctk.CTkFrame):
    def __init__(self, parent, on_back_click, on_train_click):
        super().__init__(parent, fg_color="black", corner_radius=0)

        font_arcade = ctk.CTkFont(
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
        self.form_frame.pack(fill="x", pady=10, padx=20)
        self.form_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        name_frame = ctk.CTkFrame(self, fg_color="transparent")
        name_frame.pack(padx=20, pady=(0, 10))

        name_label = ctk.CTkLabel(name_frame, text="Nazwa modelu")
        name_label.grid(row=0, column=0, padx=(0, 10))

        self.model_name_entry = ctk.CTkEntry(name_frame, width=260)
        self.model_name_entry.insert(0, "model")
        self.model_name_entry.grid(row=0, column=1)

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

        for index, (name, label_text, default_value) in enumerate(fields):
            row = index // 2
            column = (index % 2) * 2

            label = ctk.CTkLabel(
                self.form_frame,
                text=label_text,
                wraplength=180,
                justify="left",
            )

            entry = ctk.CTkEntry(
                self.form_frame,
                width=130,
            )

            entry.insert(0, default_value)

            label.grid(
                row=row,
                column=column,
                padx=10,
                pady=5,
                sticky="w",
            )

            entry.grid(
                row=row,
                column=column + 1,
                padx=10,
                pady=5,
                sticky="ew",
            )

            if algorithm == "PPO":
                CTkToolTip(
                    label,
                    message=PPO_TOOLTIPS[name],
                    delay=0.3,
                    wraplength=300,
                    justify="left",
                )

            self.entries[name] = entry

    def set_form(self, form):
        self.algorithm_select.set(form["algorithm"])
        self.update_fields(form["algorithm"])
        for name, entry in self.entries.items():
            entry.delete(0, "end")
            entry.insert(0, form[name])
        self.model_name_entry.delete(0, "end")
        self.model_name_entry.insert(0, form["model_name"])

    def get_form(self):
        form = {name: entry.get() for name, entry in self.entries.items()}
        form["algorithm"] = self.algorithm_select.get()
        form["model_name"] = self.model_name_entry.get()
        return form
