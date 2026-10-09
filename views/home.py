import customtkinter as ctk

class Home(ctk.CTkFrame):
    def __init__(self, parent, on_train_click, on_model_click, on_quit_click):
        super().__init__(parent)

        font_arcade = ctk.CTkFont(
            family="Consolas",
            size=24,
            weight="bold"
        )

        label = ctk.CTkLabel(
            self,
            text="HELLO WORLD",
            font=font_arcade
        )

        label.place(relx=0.5, rely=0.1, anchor="center")


        train_button = ctk.CTkButton(
            self,
            text="Trenuj",
            command=on_train_click
        )

        train_button.place(relx=0.5, rely=0.5, anchor="center")

        model_button = ctk.CTkButton(
            self,
            text="Wybierz model",
            command=on_model_click
        )

        model_button.place(relx=0.5, rely=0.65, anchor="center")

        quit_button = ctk.CTkButton(
            self,
            text="Wyjdź",
            command=on_quit_click
        )

        quit_button.place(relx=0.5, rely=0.80, anchor="center")
