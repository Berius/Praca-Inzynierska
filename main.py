import customtkinter as ctk

from views.home import Home
from views.train_menu import TrainMenu

ctk.set_appearance_mode("dark")

app = ctk.CTk()
app.geometry("600x700")
app.title("Praca inżynierska")

current_view = None


def change_view(view):
    global current_view

    if current_view is not None:
        current_view.destroy()

    current_view = view

    current_view.pack(
        fill="both",
        expand=True
    )


def show_train():
    change_view(TrainMenu(app))


def show_home():
    change_view(
        Home(
            app, on_train_click=show_train, on_model_click=show_train, on_quit_click=app.destroy,
        )
    )


show_home()

app.mainloop()
