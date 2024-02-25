import os
from tkinter import *

window = Tk()
window.title("Welcome")
window.geometry('1100x550')
window.configure(bg='purple')

text_label = Label(window, text="Welcome to ZenMate: Your mindful work companion!\nAre you ready to begin your journey into Zen?", fg='white', font=('Arial', 38), bg='#C8A1C8')
text_label.grid(column=0, row=0, pady=50)

def run():
    window.quit()
    os.system('python /Users/anerie/Desktop/lunahacks/main.py')

btn = Button(window, text="Ok", font=('Arial', 30), command=run)
btn.configure(bg="#4B0082")
btn.grid(column=0, row=1)

window.mainloop()
