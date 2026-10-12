import customtkinter as ctk
from pathlib import Path
from PIL import Image

class Home(ctk.CTkFrame):
    def __init__(self, parent, on_train_click, on_model_click, on_quit_click):
        super().__init__(parent, fg_color="black", corner_radius=0)

        crab_path = Path(__file__).resolve().parent.parent / "assets" / "img" / "crab.png"
        with Image.open(crab_path) as image:
            crab = image.convert("RGBA").rotate(20, expand=True)
        self.crab_image = ctk.CTkImage(light_image=crab, dark_image=crab, size=(300, 300))
        crab_label = ctk.CTkLabel(self, text="", image=self.crab_image)
        crab_label.place(relx=1, rely=1, x=80, y=80, anchor="se")

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
