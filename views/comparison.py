import customtkinter as ctk


class Comparison(ctk.CTkFrame):
    def __init__(self, parent, runs, on_compare_click, on_back_click):
        super().__init__(parent, fg_color="black", corner_radius=0)
        self.on_compare_click = on_compare_click
        self.checkboxes = {}

        title = ctk.CTkLabel(
            self, text="Porównywanie", font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(20, 10))

        run_list = ctk.CTkScrollableFrame(self, label_text="Eksperymenty")
        run_list.pack(fill="both", expand=True, padx=20, pady=10)
        for name in runs:
            checkbox = ctk.CTkCheckBox(run_list, text=name)
            checkbox.pack(anchor="w", padx=6, pady=6)
            self.checkboxes[name] = checkbox

        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.pack(pady=(6, 20))
        self.compare_button = ctk.CTkButton(
            buttons, text="Porównaj", command=self.compare,
            state="normal" if runs else "disabled",
        )
        self.compare_button.pack(side="left", padx=6)
        self.back_button = ctk.CTkButton(buttons, text="Powrót", command=on_back_click)
        self.back_button.pack(side="left", padx=6)

    def compare(self):
        selected = [name for name, checkbox in self.checkboxes.items() if checkbox.get()]
        if selected:
            self.on_compare_click(selected)
