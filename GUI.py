import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import subprocess
import os
import threading
import queue

def run_extractface():
    name = name_entry.get()
    if(name == ""):
        log_text.insert(tk.END, "Please enter the name of the sampler.\n")
        return
    laymau_path = os.path.join(os.getcwd(), "extract_face.py")
    process = subprocess.run(
        ["python", laymau_path, name],
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
    )
    if process.stdout:
        log_text.insert(tk.END, f"Output:\n{process.stdout}\n")
    if process.stderr:
        log_text.insert(tk.END, f"Errors:\n{process.stderr}\n")


def run_facerecognition():
    
    nhandien_path = os.path.join(os.getcwd(), "face_recognition.py")
    # Capture output and errors from the subprocess
    process = subprocess.run(
        ["python", nhandien_path],
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True  # Ensures output is a string
    )

    # Display the output in the GUI
    if process.stdout:
        log_text.insert(tk.END, f"Output:\n{process.stdout}\n")
    if process.stderr:
        log_text.insert(tk.END, f"Errors:\n{process.stderr}\n")
    if process.returncode == 0:
        log_text.insert(tk.END, "Face recognition completed successfully.\n")
    else:
        log_text.insert(tk.END, "Face recognition failed. Check errors above.\n")


root = tk.Tk()
root.title("Application")
root.geometry("600x500")
root.resizable(False, False)

name_label = tk.Label(root, text="Name of the sampler:", font=("Arial", 12), width=20)
name_label.pack(padx=10, pady=5)

name_entry = tk.Entry(root, width=30)
name_entry.pack(padx=10, pady=5)

laymau_button = tk.Button(root, text="Sample", command=run_extractface, width=15)
laymau_button.pack(padx=10, pady=10)

nhandien_button = tk.Button(root, text="Identification", command=run_facerecognition, width=15)
nhandien_button.pack(padx=10, pady=10)

# Log output area
log_text = ScrolledText(root, width=70, height=15, font=("Arial", 10))
log_text.pack(padx=10, pady=10)

root.mainloop()

