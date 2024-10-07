import tkinter as tk

root = tk.Tk()

root.geometry('500x500')
root.title('My First GUI')


label = tk.Label(root, text = 'Hello, World!', font = ('Helvetica', 18))

label.pack(padx = 20, pady = 20)

textbox = tk.Text(root, height = 3, font = ('Helvetica', 16))
textbox.pack(padx = 20, pady = 20)

# entry = tk.Entry(root)
# entry.pack()

button = tk.Button(root, text = 'Click Me!', font = ('Helvetica', 18))
button.pack(padx = 10, pady = 10)

root.mainloop()