import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import random
from pathlib import Path


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"


# =========================================================
# COLORS
# =========================================================

BG = "#F5F7FF"
HEADER = "#5146D8"
HEADER_DARK = "#3F36B5"

CARD = "#FFFFFF"

TEXT = "#1F2340"
SUBTEXT = "#70758A"

LETTER_BG = "#EEF0FF"
LETTER_TEXT = "#4038B5"

ACCENT = "#FFB84D"

BUTTON = "#6258E8"
BUTTON_HOVER = "#493FC7"


# =========================================================
# GLOBAL VARIABLES
# =========================================================

current_letter = ""
image_files = []
current_index = 0


# =========================================================
# HOVER EFFECT
# =========================================================

def add_hover(button):

    def mouse_enter(event):
        button.configure(
            bg=BUTTON_HOVER
        )

    def mouse_leave(event):
        button.configure(
            bg=BUTTON
        )

    button.bind(
        "<Enter>",
        mouse_enter
    )

    button.bind(
        "<Leave>",
        mouse_leave
    )


# =========================================================
# LEARN SELECTED LETTER
# =========================================================

def learn_letter(letter):

    global current_letter
    global image_files
    global current_index

    current_letter = letter

    current_index = 0

    folder = DATASET_DIR / letter

    # Check folder
    if not folder.exists():

        messagebox.showerror(
            "Letter Not Found",
            f"The dataset folder for letter {letter} was not found."
        )

        return

    # Get images
    image_files = [
        file
        for file in folder.iterdir()
        if file.suffix.lower()
        in [".jpg", ".jpeg", ".png"]
    ]

    # Check images
    if not image_files:

        messagebox.showerror(
            "No Images",
            f"No sign images were found for letter {letter}."
        )

        return

    # Start with a random example
    current_index = random.randint(
        0,
        len(image_files) - 1
    )

    show_image_window()


# =========================================================
# LARGE IMAGE WINDOW
# =========================================================

def open_large_image(parent_window):

    if not image_files:
        return

    large_window = tk.Toplevel(
        parent_window
    )

    large_window.title(
        f"ISL Letter {current_letter} - Large View"
    )

    large_window.geometry(
        "950x750"
    )

    large_window.configure(
        bg="#111827"
    )

    large_window.resizable(
        True,
        True
    )

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    top_bar = tk.Frame(
        large_window,
        bg=HEADER,
        height=80
    )

    top_bar.pack(
        fill="x"
    )

    top_bar.pack_propagate(
        False
    )

    large_title = tk.Label(
        top_bar,
        text=f"ISL Letter  {current_letter}",
        font=("Segoe UI", 24, "bold"),
        bg=HEADER,
        fg="white"
    )

    large_title.pack(
        pady=20
    )

    # -----------------------------------------------------
    # Read current image
    # -----------------------------------------------------

    image = cv2.imread(
        str(image_files[current_index])
    )

    if image is None:

        large_window.destroy()

        messagebox.showerror(
            "Image Error",
            "Unable to open this image."
        )

        return

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        image
    )

    # -----------------------------------------------------
    # Fit image
    # -----------------------------------------------------

    pil_image.thumbnail(
        (850, 580)
    )

    photo = ImageTk.PhotoImage(
        pil_image
    )

    # -----------------------------------------------------
    # Image
    # -----------------------------------------------------

    image_label = tk.Label(
        large_window,
        image=photo,
        bg="#111827",
        cursor="hand2"
    )

    image_label.image = photo

    image_label.pack(
        expand=True
    )

    # -----------------------------------------------------
    # Bottom text
    # -----------------------------------------------------

    instruction = tk.Label(
        large_window,
        text="Click the image or press ESC to close",
        font=("Segoe UI", 11),
        bg="#111827",
        fg="white"
    )

    instruction.pack(
        pady=(0, 18)
    )

    # -----------------------------------------------------
    # Close
    # -----------------------------------------------------

    image_label.bind(
        "<Button-1>",
        lambda event: large_window.destroy()
    )

    large_window.bind(
        "<Escape>",
        lambda event: large_window.destroy()
    )


# =========================================================
# IMAGE WINDOW
# =========================================================

def show_image_window():

    global current_index

    window = tk.Toplevel(
        root
    )

    window.title(
        f"Learn ISL Sign - {current_letter}"
    )

    window.geometry(
        "900x760"
    )

    window.configure(
        bg=BG
    )

    window.resizable(
        False,
        False
    )

    # =====================================================
    # HEADER
    # =====================================================

    header = tk.Frame(
        window,
        bg=HEADER,
        height=100
    )

    header.pack(
        fill="x"
    )

    header.pack_propagate(
        False
    )

    title = tk.Label(
        header,
        text=f"Learn ISL Letter  {current_letter}",
        font=("Segoe UI", 27, "bold"),
        bg=HEADER,
        fg="white"
    )

    title.pack(
        pady=27
    )

    # =====================================================
    # IMAGE CARD
    # =====================================================

    image_card = tk.Frame(
        window,
        bg=CARD,
        highlightthickness=1,
        highlightbackground="#E0E3F0"
    )

    image_card.pack(
        padx=55,
        pady=28
    )

    # =====================================================
    # IMAGE LABEL
    # =====================================================

    image_label = tk.Label(
        image_card,
        bg=CARD,
        cursor="hand2"
    )

    image_label.pack(
        padx=25,
        pady=25
    )

    # =====================================================
    # CLICK IMAGE TO ENLARGE
    # =====================================================

    image_label.bind(
        "<Button-1>",
        lambda event: open_large_image(window)
    )

    # =====================================================
    # INFORMATION
    # =====================================================

    info_label = tk.Label(
        window,
        text="Observe the hand position carefully and practice the sign.",
        font=("Segoe UI", 12),
        bg=BG,
        fg=SUBTEXT
    )

    info_label.pack(
        pady=(0, 8)
    )

    counter_label = tk.Label(
        window,
        text="",
        font=("Segoe UI", 10, "bold"),
        bg=BG,
        fg=SUBTEXT
    )

    counter_label.pack(
        pady=(0, 15)
    )

    # =====================================================
    # UPDATE IMAGE
    # =====================================================

    def update_image():

        if not image_files:
            return

        image_path = image_files[current_index]

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            return

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(
            image
        )

        # Fit image in card
        pil_image.thumbnail(
            (550, 430)
        )

        photo = ImageTk.PhotoImage(
            pil_image
        )

        image_label.configure(
            image=photo
        )

        image_label.image = photo

        counter_label.configure(
            text=f"Example {current_index + 1} of {len(image_files)}"
        )

    # =====================================================
    # NEXT IMAGE
    # =====================================================

    def next_image():

        global current_index

        current_index = (
            current_index + 1
        ) % len(image_files)

        update_image()

    # =====================================================
    # PREVIOUS IMAGE
    # =====================================================

    def previous_image():

        global current_index

        current_index = (
            current_index - 1
        ) % len(image_files)

        update_image()

    # =====================================================
    # BUTTON AREA
    # =====================================================

    button_frame = tk.Frame(
        window,
        bg=BG
    )

    button_frame.pack(
        pady=5
    )

    # Previous
    previous_button = tk.Button(
        button_frame,
        text="←  Previous",
        font=("Segoe UI", 11, "bold"),
        bg=BUTTON,
        fg="white",
        activebackground=BUTTON_HOVER,
        activeforeground="white",
        relief="flat",
        bd=0,
        cursor="hand2",
        width=16,
        pady=11,
        command=previous_image
    )

    previous_button.grid(
        row=0,
        column=0,
        padx=8
    )

    # Next
    next_button = tk.Button(
        button_frame,
        text="Next  →",
        font=("Segoe UI", 11, "bold"),
        bg=BUTTON,
        fg="white",
        activebackground=BUTTON_HOVER,
        activeforeground="white",
        relief="flat",
        bd=0,
        cursor="hand2",
        width=16,
        pady=11,
        command=next_image
    )

    next_button.grid(
        row=0,
        column=1,
        padx=8
    )

    add_hover(
        previous_button
    )

    add_hover(
        next_button
    )

    # =====================================================
    # CLOSE / BACK BUTTON
    # =====================================================

    close_button = tk.Button(
        window,
        text="←  Back to Letters",
        font=("Segoe UI", 10, "bold"),
        bg=BG,
        fg=HEADER_DARK,
        activebackground=BG,
        activeforeground=HEADER_DARK,
        relief="flat",
        bd=0,
        cursor="hand2",
        command=window.destroy
    )

    close_button.pack(
        pady=18
    )

    # Show first image
    update_image()


# =========================================================
# MAIN LEARNING WINDOW
# =========================================================

root = tk.Tk()

root.title(
    "ISL Vision - Learn A-Z"
)

root.geometry(
    "1000x720"
)

root.configure(
    bg=BG
)

root.resizable(
    False,
    False
)


# =========================================================
# HEADER
# =========================================================

header = tk.Frame(
    root,
    bg=HEADER,
    height=140
)

header.pack(
    fill="x"
)

header.pack_propagate(
    False
)


# Logo
logo = tk.Label(
    header,
    text="🤟",
    font=("Segoe UI Emoji", 42),
    bg=HEADER,
    fg="white"
)

logo.pack(
    side="left",
    padx=(55, 20)
)


# Header text
header_text = tk.Frame(
    header,
    bg=HEADER
)

header_text.pack(
    side="left",
    pady=25
)


main_title = tk.Label(
    header_text,
    text="Learn Indian Sign Language",
    font=("Segoe UI", 28, "bold"),
    bg=HEADER,
    fg="white"
)

main_title.pack(
    anchor="w"
)


main_subtitle = tk.Label(
    header_text,
    text="Explore and learn ISL signs from A to Z",
    font=("Segoe UI", 12),
    bg=HEADER,
    fg="#E5E3FF"
)

main_subtitle.pack(
    anchor="w",
    pady=4
)


# =========================================================
# INTRO
# =========================================================

intro = tk.Label(
    root,
    text="Choose a letter to learn its Indian Sign Language gesture",
    font=("Segoe UI", 14),
    bg=BG,
    fg=SUBTEXT
)

intro.pack(
    pady=(28, 20)
)


# =========================================================
# LETTER CARD
# =========================================================

letter_card = tk.Frame(
    root,
    bg=CARD,
    highlightthickness=1,
    highlightbackground="#E0E3F0"
)

letter_card.pack(
    padx=55,
    pady=5
)


# =========================================================
# A-Z BUTTONS
# =========================================================

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

for i, letter in enumerate(letters):

    button = tk.Button(
        letter_card,
        text=letter,
        font=("Segoe UI", 18, "bold"),
        bg=LETTER_BG,
        fg=LETTER_TEXT,
        activebackground=ACCENT,
        activeforeground="white",
        relief="flat",
        bd=0,
        cursor="hand2",
        width=5,
        height=2,
        command=lambda l=letter: learn_letter(l)
    )

    row = i // 7
    column = i % 7

    button.grid(
        row=row,
        column=column,
        padx=9,
        pady=9
    )

    # Hover
    def create_hover(btn):

        def enter(event):
            btn.configure(
                bg=ACCENT,
                fg="white"
            )

        def leave(event):
            btn.configure(
                bg=LETTER_BG,
                fg=LETTER_TEXT
            )

        btn.bind(
            "<Enter>",
            enter
        )

        btn.bind(
            "<Leave>",
            leave
        )

    create_hover(button)


# =========================================================
# TIP
# =========================================================

tip = tk.Label(
    root,
    text="💡 Tip: Click a letter and observe the sign carefully.",
    font=("Segoe UI", 11),
    bg=BG,
    fg=SUBTEXT
)

tip.pack(
    pady=20
)


# =========================================================
# BACK TO HOME
# =========================================================

back_button = tk.Button(
    root,
    text="←  Back to Home",
    font=("Segoe UI", 11, "bold"),
    bg=HEADER,
    fg="white",
    activebackground=HEADER_DARK,
    activeforeground="white",
    relief="flat",
    bd=0,
    cursor="hand2",
    width=20,
    pady=10,
    command=root.destroy
)

back_button.pack(
    pady=5
)

add_hover(
    back_button
)


# =========================================================
# START
# =========================================================

root.mainloop()