import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
from pathlib import Path


# =========================================================
# PROJECT PATH
# =========================================================

CODE_DIR = Path(__file__).resolve().parent


# =========================================================
# COLORS
# =========================================================

BG = "#F4F7FB"
CARD = "#FFFFFF"
PRIMARY = "#3155D4"
PRIMARY_DARK = "#2443AE"
TEXT = "#172033"
SECONDARY = "#667085"
LIGHT_BLUE = "#E9EEFF"
DANGER = "#D92D20"


# =========================================================
# RUN PROGRAM
# =========================================================

def run_program(filename):

    program = CODE_DIR / filename

    try:
        subprocess.Popen(
            [sys.executable, str(program)]
        )

    except Exception as error:

        messagebox.showerror(
            "Error",
            f"Unable to open the selected feature.\n\n{error}"
        )


# =========================================================
# BUTTON FUNCTIONS
# =========================================================

def open_recognition():

    run_program("recognize_letters.py")


def open_learning():

    run_program("learning_mode.py")


def exit_application():

    answer = messagebox.askyesno(
        "Exit Application",
        "Are you sure you want to exit?"
    )

    if answer:
        root.destroy()


# =========================================================
# MAIN WINDOW
# =========================================================

root = tk.Tk()

root.title(
    "ISL Vision - Indian Sign Language"
)

root.geometry("1000x650")

root.minsize(900, 600)

root.configure(
    bg=BG
)


# =========================================================
# HEADER
# =========================================================

header = tk.Frame(
    root,
    bg=PRIMARY,
    height=125
)

header.pack(
    fill="x"
)

header.pack_propagate(False)


# Logo / Symbol

logo = tk.Label(
    header,
    text="🤟",
    font=("Segoe UI Emoji", 38),
    bg=PRIMARY,
    fg="white"
)

logo.pack(
    side="left",
    padx=(45, 15)
)


# Header text

header_text = tk.Frame(
    header,
    bg=PRIMARY
)

header_text.pack(
    side="left",
    pady=20
)


title = tk.Label(
    header_text,
    text="ISL VISION",
    font=("Segoe UI", 27, "bold"),
    bg=PRIMARY,
    fg="white"
)

title.pack(
    anchor="w"
)


subtitle = tk.Label(
    header_text,
    text="Indian Sign Language • Learn & Recognize",
    font=("Segoe UI", 12),
    bg=PRIMARY,
    fg="#DCE4FF"
)

subtitle.pack(
    anchor="w",
    pady=(3, 0)
)


# =========================================================
# MAIN CONTENT
# =========================================================

content = tk.Frame(
    root,
    bg=BG
)

content.pack(
    fill="both",
    expand=True,
    padx=55,
    pady=35
)


# Welcome section

welcome = tk.Label(
    content,
    text="Welcome 👋",
    font=("Segoe UI", 24, "bold"),
    bg=BG,
    fg=TEXT
)

welcome.pack(
    anchor="w"
)


description = tk.Label(
    content,
    text="Explore Indian Sign Language through interactive recognition and learning.",
    font=("Segoe UI", 12),
    bg=BG,
    fg=SECONDARY
)

description.pack(
    anchor="w",
    pady=(5, 30)
)


# =========================================================
# CARDS AREA
# =========================================================

cards = tk.Frame(
    content,
    bg=BG
)

cards.pack(
    fill="x"
)


# =========================================================
# RECOGNITION CARD
# =========================================================

recognition_card = tk.Frame(
    cards,
    bg=CARD,
    width=410,
    height=280,
    highlightthickness=1,
    highlightbackground="#E4E7EC"
)

recognition_card.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 15)
)

recognition_card.pack_propagate(False)


# Icon

recognition_icon = tk.Label(
    recognition_card,
    text="📷",
    font=("Segoe UI Emoji", 38),
    bg=CARD,
    fg=PRIMARY
)

recognition_icon.pack(
    pady=(25, 5)
)


recognition_title = tk.Label(
    recognition_card,
    text="A–Z Letter Recognition",
    font=("Segoe UI", 17, "bold"),
    bg=CARD,
    fg=TEXT
)

recognition_title.pack()


recognition_description = tk.Label(
    recognition_card,
    text="Use the camera to recognize\nIndian Sign Language letters.",
    font=("Segoe UI", 11),
    bg=CARD,
    fg=SECONDARY,
    justify="center"
)

recognition_description.pack(
    pady=12
)


recognition_button = tk.Button(
    recognition_card,
    text="START RECOGNITION  →",
    font=("Segoe UI", 11, "bold"),
    bg=PRIMARY,
    fg="white",
    activebackground=PRIMARY_DARK,
    activeforeground="white",
    relief="flat",
    bd=0,
    padx=20,
    pady=10,
    cursor="hand2",
    command=open_recognition
)

recognition_button.pack(
    pady=8
)


# =========================================================
# LEARNING CARD
# =========================================================

learning_card = tk.Frame(
    cards,
    bg=CARD,
    width=410,
    height=280,
    highlightthickness=1,
    highlightbackground="#E4E7EC"
)

learning_card.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(15, 0)
)

learning_card.pack_propagate(False)


# Icon

learning_icon = tk.Label(
    learning_card,
    text="📚",
    font=("Segoe UI Emoji", 38),
    bg=CARD,
    fg=PRIMARY
)

learning_icon.pack(
    pady=(25, 5)
)


learning_title = tk.Label(
    learning_card,
    text="Learn A–Z Signs",
    font=("Segoe UI", 17, "bold"),
    bg=CARD,
    fg=TEXT
)

learning_title.pack()


learning_description = tk.Label(
    learning_card,
    text="Explore ISL alphabet signs and\nlearn them using example images.",
    font=("Segoe UI", 11),
    bg=CARD,
    fg=SECONDARY,
    justify="center"
)

learning_description.pack(
    pady=12
)


learning_button = tk.Button(
    learning_card,
    text="START LEARNING  →",
    font=("Segoe UI", 11, "bold"),
    bg=PRIMARY,
    fg="white",
    activebackground=PRIMARY_DARK,
    activeforeground="white",
    relief="flat",
    bd=0,
    padx=20,
    pady=10,
    cursor="hand2",
    command=open_learning
)

learning_button.pack(
    pady=8
)


# =========================================================
# FOOTER
# =========================================================

footer = tk.Frame(
    root,
    bg=BG
)

footer.pack(
    fill="x",
    padx=55,
    pady=(0, 25)
)


footer_text = tk.Label(
    footer,
    text="ISL Vision  •  A–Z Recognition & Learning",
    font=("Segoe UI", 9),
    bg=BG,
    fg=SECONDARY
)

footer_text.pack(
    side="left"
)


exit_button = tk.Button(
    footer,
    text="Exit",
    font=("Segoe UI", 10, "bold"),
    bg=BG,
    fg=DANGER,
    activebackground=BG,
    activeforeground=DANGER,
    relief="flat",
    bd=0,
    cursor="hand2",
    command=exit_application
)

exit_button.pack(
    side="right"
)


# =========================================================
# START APPLICATION
# =========================================================

root.mainloop()