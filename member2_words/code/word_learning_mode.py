import cv2
import tkinter as tk
from tkinter import messagebox
from pathlib import Path


# =========================================================
# PROJECT LOCATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"


# =========================================================
# GET AVAILABLE WORDS
# =========================================================

words = sorted([
    folder.name
    for folder in DATASET_DIR.iterdir()
    if folder.is_dir()
])


# =========================================================
# MAIN WINDOW
# =========================================================

root = tk.Tk()

root.title("ISL Vision - Learn Words")

root.geometry("900x700")

root.minsize(800, 600)

root.configure(
    bg="#f4f6fb"
)


# =========================================================
# CURRENT WORD
# =========================================================

current_index = 0


# =========================================================
# PLAY WORD VIDEO
# =========================================================

def show_word(word):

    word_folder = DATASET_DIR / word

    videos = sorted(
        word_folder.glob("*.mp4")
    )

    if not videos:

        messagebox.showerror(
            "Video Not Found",
            f"No ISL video found for '{word}'."
        )

        return


    video_path = str(videos[0])

    cap = cv2.VideoCapture(video_path)


    if not cap.isOpened():

        messagebox.showerror(
            "Error",
            "The ISL video could not be opened."
        )

        return


    while True:

        success, frame = cap.read()

        if not success:
            break


        frame = cv2.resize(
            frame,
            (800, 600)
        )


        cv2.putText(
            frame,
            f"ISL SIGN: {word.upper()}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            "Press Q to close video",
            (20, 580),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        cv2.imshow(
            "ISL Vision - Word Learning",
            frame
        )


        key = cv2.waitKey(30) & 0xFF


        if key == ord("q"):

            break


    cap.release()

    cv2.destroyAllWindows()


# =========================================================
# SELECT WORD
# =========================================================

def select_word(word):

    global current_index

    current_index = words.index(word)

    selected_word_label.config(
        text=word.upper()
    )


# =========================================================
# SEARCH WORD
# =========================================================

def search_word():

    query = search_entry.get().strip().lower()


    if not query:

        messagebox.showinfo(
            "Search",
            "Please enter a word."
        )

        return


    matches = [
        word
        for word in words
        if query in word.lower()
    ]


    if not matches:

        messagebox.showinfo(
            "Word Not Found",
            f"'{query}' is not available in the dataset."
        )

        return


    select_word(matches[0])


# =========================================================
# PLAY SELECTED WORD
# =========================================================

def play_selected():

    if not words:
        return

    word = words[current_index]

    show_word(word)


# =========================================================
# NEXT WORD
# =========================================================

def next_word():

    global current_index

    if not words:
        return


    if current_index < len(words) - 1:

        current_index += 1

        selected_word_label.config(
            text=words[current_index].upper()
        )


# =========================================================
# PREVIOUS WORD
# =========================================================

def previous_word():

    global current_index

    if not words:
        return


    if current_index > 0:

        current_index -= 1

        selected_word_label.config(
            text=words[current_index].upper()
        )


# =========================================================
# BACK TO MAIN MENU
# =========================================================

def go_back_to_main():

    root.destroy()


# =========================================================
# HEADER
# =========================================================

header = tk.Frame(
    root,
    bg="#29254c",
    height=130
)

header.pack(
    fill="x"
)

header.pack_propagate(False)


title = tk.Label(
    header,
    text="ISL VISION",
    font=("Arial", 30, "bold"),
    bg="#29254c",
    fg="white"
)

title.pack(
    pady=(25, 5)
)


subtitle = tk.Label(
    header,
    text="Learn Indian Sign Language Words",
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
    pady=25
)


# =========================================================
# SEARCH SECTION
# =========================================================

search_frame = tk.Frame(
    content,
    bg="#f4f6fb"
)

search_frame.pack(
    pady=10
)


search_label = tk.Label(
    search_frame,
    text="Search Word:",
    font=("Arial", 12, "bold"),
    bg="#f4f6fb",
    fg="#333344"
)

search_label.pack(
    side="left",
    padx=5
)


search_entry = tk.Entry(
    search_frame,
    font=("Arial", 13),
    width=30
)

search_entry.pack(
    side="left",
    padx=8
)


search_button = tk.Button(
    search_frame,
    text="Search",
    command=search_word,
    font=("Arial", 11, "bold"),
    bg="#5046a5",
    fg="white",
    activebackground="#3d3685",
    activeforeground="white",
    relief="flat",
    cursor="hand2",
    padx=18,
    pady=7
)

search_button.pack(
    side="left",
    padx=5
)


# =========================================================
# SELECTED WORD
# =========================================================

selected_word_label = tk.Label(
    content,
    text=words[0].upper() if words else "NO WORDS",
    font=("Arial", 30, "bold"),
    bg="white",
    fg="#5046a5",
    width=25,
    height=2
)

selected_word_label.pack(
    pady=20
)


# =========================================================
# WATCH VIDEO BUTTON
# =========================================================

play_button = tk.Button(
    content,
    text="▶  WATCH ISL SIGN",
    command=play_selected,
    font=("Arial", 14, "bold"),
    bg="#5046a5",
    fg="white",
    activebackground="#3d3685",
    activeforeground="white",
    relief="flat",
    cursor="hand2",
    padx=30,
    pady=12
)

play_button.pack(
    pady=8
)


# =========================================================
# PREVIOUS / NEXT
# =========================================================

navigation_frame = tk.Frame(
    content,
    bg="#f4f6fb"
)

navigation_frame.pack(
    pady=15
)


previous_button = tk.Button(
    navigation_frame,
    text="← Previous",
    command=previous_word,
    font=("Arial", 11, "bold"),
    bg="white",
    fg="#29254c",
    relief="flat",
    cursor="hand2",
    padx=20,
    pady=8
)

previous_button.pack(
    side="left",
    padx=10
)


next_button = tk.Button(
    navigation_frame,
    text="Next →",
    command=next_word,
    font=("Arial", 11, "bold"),
    bg="white",
    fg="#29254c",
    relief="flat",
    cursor="hand2",
    padx=20,
    pady=8
)

next_button.pack(
    side="left",
    padx=10
)


# =========================================================
# WORD COUNT
# =========================================================

count_label = tk.Label(
    content,
    text=f"Available words: {len(words)}",
    font=("Arial", 11),
    bg="#f4f6fb",
    fg="#777788"
)

count_label.pack(
    pady=5
)


# =========================================================
# INSTRUCTION
# =========================================================

instruction_label = tk.Label(
    content,
    text="Search a word and watch its ISL sign.",
    font=("Arial", 11),
    bg="#f4f6fb",
    fg="#666677"
)

instruction_label.pack(
    pady=5
)


# =========================================================
# BACK BUTTON
# =========================================================

back_button = tk.Button(
    content,
    text="← Back to Main Menu",
    command=go_back_to_main,
    font=("Arial", 11, "bold"),
    bg="#29254c",
    fg="white",
    activebackground="#3d3866",
    activeforeground="white",
    relief="flat",
    cursor="hand2",
    padx=22,
    pady=9
)

back_button.pack(
    pady=15
)


# =========================================================
# START
# =========================================================

root.mainloop()