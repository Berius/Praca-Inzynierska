import customtkinter as ctk


class ModelSelection(ctk.CTkFrame):
    def __init__(self, parent, models, on_play_click, on_back_click):
        super().__init__(parent, fg_color="black", corner_radius=0)

        title = ctk.CTkLabel(
            self,
            text="Wybierz model",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        title.pack(pady=(20, 10))

        self.model_select = ctk.CTkComboBox(
            self,
            values=models,
            state="readonly",
            width=400,
        )
        self.model_select.pack(pady=20)
        self.model_select.set(models[0] if models else "Brak zapisanych modeli")

        self.play_button = ctk.CTkButton(
            self,
            text="Uruchom grę",
            command=lambda: on_play_click(self.model_select.get()),
            state="normal" if models else "disabled",
        )
        self.play_button.pack(pady=10)

        back_button = ctk.CTkButton(
            self,
            text="Powrót",
            command=on_back_click,
        )
        back_button.pack(side="bottom", pady=20)
