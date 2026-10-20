import customtkinter as ctk


class TrainingProgress(ctk.CTkFrame):
    def __init__(self, parent, total_timesteps, progress_queue, stop_event, on_back_click):
        super().__init__(parent, fg_color="black", corner_radius=0)
        self.total_timesteps = total_timesteps
        self.progress_queue = progress_queue
        self.stop_event = stop_event
        self.on_back_click = on_back_click

        self.progress_label = ctk.CTkLabel(self, text="")
        self.progress_label.place(relx=0.5, rely=0.5, y=-30, anchor="center")

        self.progress_bar = ctk.CTkProgressBar(self, width=400)
        self.progress_bar.place(relx=0.5, rely=0.5, anchor="center")
        self.set_progress(0)

        self.stop_button = ctk.CTkButton(self, text="Zakończ", command=self.stop_training)
        self.stop_button.place(relx=0.5, rely=0.5, y=55, anchor="center")

        self.after(100, self.update_progress)

    def stop_training(self):
        self.stop_event.set()
        self.stop_button.configure(text="Kończenie...", state="disabled")

    def set_progress(self, current):
        progress = min(current / self.total_timesteps, 1)
        self.progress_bar.set(progress)
        steps = f"{current:,} / {self.total_timesteps:,}".replace(",", " ")
        percent = f"{progress * 100:.1f}".replace(".", ",")
        self.progress_label.configure(text=f"{steps} kroków ({percent}%)")

    def update_progress(self):
        while not self.progress_queue.empty():
            event, current = self.progress_queue.get()
            if event == "progress":
                self.set_progress(current)
            elif event == "done":
                self.progress_bar.set(1)
                if self.stop_event.is_set():
                    self.on_back_click()
                else:
                    self.stop_button.configure(text="Powrót", command=self.on_back_click)
                return
            elif event == "stopped":
                self.on_back_click()
                return

        self.after(100, self.update_progress)
