import customtkinter as ctk

from control.train_controller import TrainController
from control.model_selection_controller import ModelSelectionController
from control.comparison_controller import ComparisonController
from views.comparison import Comparison
from views.home import Home
from views.model_selection import ModelSelection
from views.train_menu import TrainMenu
from views.training_progress import TrainingProgress

ctk.set_appearance_mode("dark")
ctk.ThemeManager.theme["CTkFont"]["family"] = "Consolas"

app = ctk.CTk(fg_color="black")
app.geometry("800x650")
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


def show_train(form=None):
    controller = TrainController(show_training_progress)
    view = TrainMenu(
        app,
        on_back_click=show_home,
        on_train_click=controller.start_training,
    )
    if form is not None:
        view.set_form(form)
    change_view(view)


def show_training_progress(form, progress_queue, stop_event):
    change_view(
        TrainingProgress(
            app,
            int(form["total_timesteps"]),
            progress_queue,
            stop_event,
            on_back_click=lambda: show_train(form),
        )
    )


def show_model_selection():
    controller = ModelSelectionController()
    change_view(
        ModelSelection(
            app,
            models=controller.get_models(),
            on_play_click=controller.play,
            on_back_click=show_home,
        )
    )


def show_comparison():
    controller = ComparisonController()
    change_view(
        Comparison(
            app,
            runs=controller.get_runs(),
            on_compare_click=controller.compare,
            on_back_click=show_home,
        )
    )


def show_home():
    change_view(
        Home(
            app,
            on_train_click=show_train,
            on_model_click=show_model_selection,
            on_compare_click=show_comparison,
            on_quit_click=app.destroy,
        )
    )


show_home()

app.mainloop()
