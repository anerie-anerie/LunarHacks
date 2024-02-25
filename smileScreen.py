import os
from tkinter import *

window = Tk()
window.title("Break Reminder")
window.geometry('650x250')

# Add text label
text_label = Label(window, text="You've been frowning for a while, maybe it's time to take a break.\n Do something you enjoy!", fg='white', font=('Arial', 20), bg='#C8A1C8')
text_label.grid(column=0, row=0, pady=50)

def run():
    window.quit()
    os.system('python /Users/anerie/Desktop/lunahacks/main.py')

#Button
btn = Button(window, text="Ok", fg="black", font=('Arial', 12), command=run)
btn.configure(bg="#4B0082")
btn.grid(column=0, row=1)

window.mainloop()
