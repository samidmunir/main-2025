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

# button = tk.Button(root, text = 'Click Me!', font = ('Helvetica', 18))
# button.pack(padx = 10, pady = 10)

buttonFrame = tk.Frame(root)
buttonFrame.columnconfigure(0, weight = 1)
buttonFrame.columnconfigure(1, weight = 1)
buttonFrame.columnconfigure(2, weight = 1)

button1 = tk.Button(buttonFrame, text = '1', font = ('Helvetica', 18))
button1.grid(row = 0, column = 0, sticky = tk.W + tk.E)

button2 = tk.Button(buttonFrame, text = '2', font = ('Helvetica', 18))
button2.grid(row = 0, column = 1, sticky = tk.W + tk.E)

button3 = tk.Button(buttonFrame, text = '3', font = ('Helvetica', 18))
button3.grid(row = 0, column = 2, sticky = tk.W + tk.E)

button4 = tk.Button(buttonFrame, text = '4', font = ('Helvetica', 18))
button4.grid(row = 1, column = 0, sticky = tk.W + tk.E)

button5 = tk.Button(buttonFrame, text = '5', font = ('Helvetica', 18))
button5.grid(row = 1, column = 1, sticky = tk.W + tk.E)

button6 = tk.Button(buttonFrame, text = '6', font = ('Helvetica', 18))
button6.grid(row = 1, column = 2, sticky = tk.W + tk.E)

buttonFrame.pack(fill = 'x')

root.mainloop()