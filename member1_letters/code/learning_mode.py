import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

# Find all letter folders
letters = sorted([
    folder.name
    for folder in DATASET_DIR.iterdir()
    if folder.is_dir()
])

root = tk.Tk()

root.title("ISL Vision - Learn A-Z")
root.geometry("900x700")
root.minsize(800, 600)
root.configure(bg="#f4f6fb")

current_index = 0
current_image = None


# ---------------------------------
# FIND LETTER IMAGES
# ---------------------------------

def find_letter_images(letter):

    letter_folder = DATASET_DIR / letter

    image_files = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG"
    ]:
        image_files.extend(
            letter_folder.glob(extension)
        )

    return sorted(image_files)


# ---------------------------------
# SHOW LETTER IMAGE
# ---------------------------------

def show_letter(letter):

    global current_image

    image_files = find_letter_images(letter)

    if not image_files:

        image_label.config(
            image="",
            text=f"No image found for {letter.upper()}",
            font=("Arial", 18, "bold"),
            fg="#777777"
        )

        return

    image_path = image_files[0]

    try:

        image = Image.open(image_path)

        image.thumbnail((500, 350))

        current_image = ImageTk.PhotoImage(image)

        image_label.config(
            image=current_image,
            text=""
        )

    except Exception as error:

        messagebox.showerror(
            "Image Error",
            f"Could not open the image.\n\n{error}"
        )


# ---------------------------------
# SELECT LETTER
# ---------------------------------

def select_letter(letter):

    global current_index

    if letter not in letters:
        return

    current_index = letters.index(letter)

    selected_letter_label.config(
        text=letter.upper()
    )

    show_letter(letter)


# ---------------------------------
# NEXT LETTER
# ---------------------------------

def next_letter():

    global current_index

    if not letters:
        return

    if current_index < len(letters) - 1:

        current_index += 1

        letter = letters[current_index]

        selected_letter_label.config(
            text=letter.upper()
        )

        show_letter(letter)


# ---------------------------------
# PREVIOUS LETTER
# ---------------------------------

def previous_letter():

    global current_index

    if not letters:
        return

    if current_index > 0:

        current_index -= 1

        letter = letters[current_index]

        selected_letter_label.config(
            text=letter.upper()
        )

        show_letter(letter)


# ---------------------------------
# SEARCH LETTER
# ---------------------------------

def search_letter():

    query = search_entry.get().strip().upper()

    if not query:

        messagebox.showinfo(
            "Search",
            "Please enter a letter."
        )

        return

    if query in letters:

        select_letter(query)

    else:

        messagebox.showinfo(
            "Letter Not Found",
            f"Letter '{query}' is not available."
        )


# ---------------------------------
# BACK TO MAIN MENU
# ---------------------------------

def go_back_to_main():

    root.destroy()


# ---------------------------------
# HEADER
# ---------------------------------

header = tk.Frame(
    root,
    bg="#29254c",
    height=130
)

header.pack(
    fill="x"
)

header.pack_propagate(False)


# ---------------------------------
# TITLE
# ---------------------------------

title = tk.Label(
    header,
    text="ISL VISION",
    font=("Arial", 30, "bold"),
    bg="#29254c",
    fg="white"
)

title.pack(
    pady=(25, 3)
)


# ---------------------------------
# SUBTITLE
# ---------------------------------

subtitle = tk.Label(
    header,
    text="Learn Indian Sign Language A-Z",
    font=("Arial", 14),
    bg="#29254c",
    fg="#dddddd"
)

subtitle.pack()


# ---------------------------------
# BACK BUTTON
# ---------------------------------

back_button = tk.Button(
    header,
    text="← Back to Main Menu",
    command=go_back_to_main,
    font=("Arial", 10, "bold"),
    bg="#5046a5",
    fg="white",
    activebackground="#3d3685",
    activeforeground="white",
    relief="flat",
    cursor="hand2",
    padx=15,
    pady=7
)

back_button.place(
    relx=0.97,
    rely=0.5,
    anchor="e"
)


# ---------------------------------
# MAIN CONTENT
# ---------------------------------

content = tk.Frame(
    root,
    bg="#f4f6fb"
)

content.pack(
    fill="both",
    expand=True,
    padx=50,
    pady=15
)


# ---------------------------------
# SEARCH AREA
# ---------------------------------

search_frame = tk.Frame(
    content,
    bg="#f4f6fb"
)

search_frame.pack(
    pady=5
)


search_label = tk.Label(
    search_frame,
    text="Search Letter:",
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
    width=15
)

search_entry.pack(
    side="left",
    padx=8
)


search_button = tk.Button(
    search_frame,
    text="Search",
    command=search_letter,
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


# ---------------------------------
# SELECTED LETTER
# ---------------------------------

selected_letter_label = tk.Label(
    content,
    text=letters[0].upper() if letters else "NO LETTERS",
    font=("Arial", 32, "bold"),
    bg="white",
    fg="#5046a5",
    width=12,
    height=2
)

selected_letter_label.pack(
    pady=8
)


# ---------------------------------
# IMAGE AREA
# ---------------------------------

image_frame = tk.Frame(
    content,
    bg="white",
    width=520,
    height=330
)

image_frame.pack(
    pady=5
)

image_frame.pack_propagate(False)


image_label = tk.Label(
    image_frame,
    bg="white",
    fg="#777777"
)

image_label.pack(
    expand=True
)


# ---------------------------------
# NAVIGATION
# ---------------------------------

navigation_frame = tk.Frame(
    content,
    bg="#f4f6fb"
)

navigation_frame.pack(
    pady=8
)


previous_button = tk.Button(
    navigation_frame,
    text="← Previous",
    command=previous_letter,
    font=("Arial", 11, "bold"),
    bg="white",
    fg="#29254c",
    activebackground="#eeeeee",
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
    command=next_letter,
    font=("Arial", 11, "bold"),
    bg="white",
    fg="#29254c",
    activebackground="#eeeeee",
    relief="flat",
    cursor="hand2",
    padx=20,
    pady=8
)

next_button.pack(
    side="left",
    padx=10
)


# ---------------------------------
# INFORMATION
# ---------------------------------

count_label = tk.Label(
    content,
    text=f"Available letters: {len(letters)}",
    font=("Arial", 10),
    bg="#f4f6fb",
    fg="#777788"
)

count_label.pack(
    pady=2
)


instruction_label = tk.Label(
    content,
    text="Search a letter or use Previous / Next to learn ISL signs.",
    font=("Arial", 10),
    bg="#f4f6fb",
    fg="#666677"
)

instruction_label.pack(
    pady=2
)


# ---------------------------------
# SHOW FIRST LETTER
# ---------------------------------

if letters:

    show_letter(
        letters[current_index]
    )


# ---------------------------------
# START APPLICATION
# ---------------------------------

root.mainloop()