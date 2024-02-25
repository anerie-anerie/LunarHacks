import os
from tkinter import *

window = Tk()
window.title("Break Reminder")
window.geometry('320x250')

# Add text label
text_label = Label(window, text="Take a break!\nAre you ready to come back?", fg='white', font=('Arial', 24), bg='#C8A1C8')
text_label.grid(column=0, row=0, pady=50)

def run():
    window.quit()
    os.system('python /Users/anerie/Desktop/lunahacks/main.py')

# Button
btn = Button(window, text="Ok", fg="black", font=('Arial', 12), command=run)
btn.configure(bg="#4B0082")
btn.grid(column=0, row=1)

window.mainloop()
