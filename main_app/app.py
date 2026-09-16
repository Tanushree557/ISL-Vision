import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
from pathlib import Path


# =========================================================
# PROJECT LOCATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# =========================================================
# MEMBER 1 - LETTERS
# =========================================================

LETTER_RECOGNITION = (
    BASE_DIR
    / "member1_letters"
    / "code"
    / "recognize_letters.py"
)

LETTER_LEARNING = (
    BASE_DIR
    / "member1_letters"
    / "code"
    / "learning_mode.py"
)


# =========================================================
# MEMBER 2 - WORDS
# =========================================================

WORD_RECOGNITION = (
    BASE_DIR
    / "member2_words"
    / "code"
    / "recognize_words.py"
)

WORD_LEARNING = (
    BASE_DIR
    / "member2_words"
    / "code"
    / "word_learning_mode.py"
)


# =========================================================
# RUN PROGRAM
# =========================================================

def run_program(file_path):

    if not file_path.exists():

        messagebox.showerror(
            "File Not Found",
            f"Could not find:\n\n{file_path}"
        )

        return

    try:

        subprocess.Popen(
            [
                sys.executable,
                str(file_path)
            ],
            cwd=str(file_path.parent)
        )

    except Exception as error:

        messagebox.showerror(
            "Error",
            f"Could not start the program.\n\n{error}"
        )


# =========================================================
# LETTER FUNCTIONS
# =========================================================

def open_letter_recognition():

    run_program(LETTER_RECOGNITION)


def open_letter_learning():

    run_program(LETTER_LEARNING)


# =========================================================
# WORD FUNCTIONS
# =========================================================

def open_word_recognition():

    run_program(WORD_RECOGNITION)


def open_word_learning():

    run_program(WORD_LEARNING)


# =========================================================
# EXIT
# =========================================================

def exit_application():

    answer = messagebox.askyesno(
        "Exit ISL Vision",
        "Do you want to exit ISL Vision?"
    )

    if answer:

        root.destroy()


# =========================================================
# MAIN WINDOW
# =========================================================

root = tk.Tk()

root.title(
    "ISL Vision - Indian Sign Language Assistant"
)

root.geometry(
    "900x650"
)

root.minsize(
    800,
    600
)

root.configure(
    bg="#f4f6fb"
)


# =========================================================
# HEADER
# =========================================================

header = tk.Frame(
    root,
    bg="#29254c",
    height=140
)

header.pack(
    fill="x"
)

header.pack_propagate(False)


title = tk.Label(
    header,
    text="ISL VISION",
    font=("Arial", 32, "bold"),
    bg="#29254c",
    fg="white"
)

title.pack(
    pady=(28, 5)
)


subtitle = tk.Label(
    header,
    text="Indian Sign Language Assistant",
    font=("Arial", 14),
    bg="#29254c",
    fg="#dddddd"
)

subtitle.pack()


# =========================================================
# MAIN CONTENT
# =========================================================

content = tk.Frame(
    root,
    bg="#f4f6fb"
)

content.pack(
    fill="both",
    expand=True,
    padx=50,
    pady=30
)


# =========================================================
# LETTERS
# =========================================================

letters_frame = tk.LabelFrame(
    content,
    text="  ✋ LETTERS  ",
    font=("Arial", 16, "bold"),
    bg="white",
    fg="#5046a5",
    padx=25,
    pady=20
)

letters_frame.pack(
    fill="x",
    pady=10
)


letters_info = tk.Label(
    letters_frame,
    text="Recognize and learn ISL alphabet signs A-Z",
    font=("Arial", 10),
    bg="white",
    fg="#777777"
)

letters_info.pack(
    pady=(0, 15)
)


letters_buttons = tk.Frame(
    letters_frame,
    bg="white"
)

letters_buttons.pack()


letter_recognition_button = tk.Button(
    letters_buttons,
    text="Letter Recognition",
    command=open_letter_recognition,
    font=("Arial", 12, "bold"),
    bg="#eeeafd",
    fg="#29254c",
    activebackground="#ddd6ff",
    relief="flat",
    cursor="hand2",
    width=22,
    height=2
)

letter_recognition_button.pack(
    side="left",
    padx=10
)


letter_learning_button = tk.Button(
    letters_buttons,
    text="Learn A-Z Signs",
    command=open_letter_learning,
    font=("Arial", 12, "bold"),
    bg="#eeeafd",
    fg="#29254c",
    activebackground="#ddd6ff",
    relief="flat",
    cursor="hand2",
    width=22,
    height=2
)

letter_learning_button.pack(
    side="left",
    padx=10
)


# =========================================================
# WORDS
# =========================================================

words_frame = tk.LabelFrame(
    content,
    text="  🤟 WORDS  ",
    font=("Arial", 16, "bold"),
    bg="white",
    fg="#5046a5",
    padx=25,
    pady=20
)

words_frame.pack(
    fill="x",
    pady=10
)


words_info = tk.Label(
    words_frame,
    text="Recognize and learn ISL word signs",
    font=("Arial", 10),
    bg="white",
    fg="#777777"
)

words_info.pack(
    pady=(0, 15)
)


words_buttons = tk.Frame(
    words_frame,
    bg="white"
)

words_buttons.pack()


word_recognition_button = tk.Button(
    words_buttons,
    text="Word Recognition",
    command=open_word_recognition,
    font=("Arial", 12, "bold"),
    bg="#eeeafd",
    fg="#29254c",
    activebackground="#ddd6ff",
    relief="flat",
    cursor="hand2",
    width=22,
    height=2
)

word_recognition_button.pack(
    side="left",
    padx=10
)


word_learning_button = tk.Button(
    words_buttons,
    text="Learn Words",
    command=open_word_learning,
    font=("Arial", 12, "bold"),
    bg="#eeeafd",
    fg="#29254c",
    activebackground="#ddd6ff",
    relief="flat",
    cursor="hand2",
    width=22,
    height=2
)

word_learning_button.pack(
    side="left",
    padx=10
)


# =========================================================
# EXIT
# =========================================================

exit_button = tk.Button(
    content,
    text="EXIT",
    command=exit_application,
    font=("Arial", 12, "bold"),
    bg="#29254c",
    fg="white",
    activebackground="#3d3866",
    activeforeground="white",
    relief="flat",
    cursor="hand2",
    width=18,
    height=2
)

exit_button.pack(
    pady=25
)


# =========================================================
# FOOTER
# =========================================================

footer = tk.Label(
    root,
    text="ISL Vision • AI-based Sign Language Learning & Recognition",
    font=("Arial", 9),
    bg="#f4f6fb",
    fg="#888899"
)

footer.pack(
    pady=(0, 12)
)


# =========================================================
# START
# =========================================================

root.mainloop()