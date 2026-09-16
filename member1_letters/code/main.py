import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CODE_DIR = BASE_DIR


def run_program(filename):
    program = CODE_DIR / filename
    subprocess.run([sys.executable, str(program)])


while True:

    print("\n==============================")
    print("   ISL LETTER LEARNING APP")
    print("==============================")
    print("1. A-Z Letter Recognition")
    print("2. A-Z Learning Mode")
    print("3. Exit")
    print("==============================")

    choice = input("Enter your choice: ")

    if choice == "1":
        run_program("recognize_letters.py")

    elif choice == "2":
        run_program("learning_mode.py")

    elif choice == "3":
        print("Application closed.")
        break

    else:
        print("Invalid choice. Please enter 1, 2, or 3.")