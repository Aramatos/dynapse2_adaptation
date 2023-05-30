import tkinter as tk
import json
import subprocess

# Define the JSON file name
json_file = 'test_select.json'

# Load the initial JSON data
with open(json_file, 'r') as f:
    data = json.load(f)

# Define a function to update the JSON data based on the button values
def update_data():
    data[var.get()] = True
    
    for key in data.keys():
        if key != var.get():
            data[key] = False
    
    # Write the updated data to the JSON file
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)

def start_dynapse_2():
    cmd = ['python', 'main.py', "./bitfiles/Dynapse2Stack.bit", "1"]
    subprocess.run(cmd)


# Define a function to handle window close events
def on_close():
    root.destroy()

# Create the main window
root = tk.Tk()
root.title("Test Selection")

# Set the background color
root.configure(bg='#333333')

# Set the font
font = ('Courier', 14, 'bold')

# Set the button style
button_style = {
    'bg': '#FFD700',
    'fg': '#333333',
    'activebackground': '#333333',
    'activeforeground': '#FFD700',
    'bd': 2,
    'highlightthickness': 0,
    'relief': 'flat',
    'font': font,
}

# Create the radio buttons
var = tk.StringVar()
var.set('FF_Single_Neurons')
button1 = tk.Radiobutton(root, text="FF_Network", variable=var, value='FF_Network', command=update_data, **button_style)
button1.grid(row=0, column=0, padx=10, pady=10)

button2 = tk.Radiobutton(root, text="FF_PC_PV", variable=var, value='FF_PC_PV', command=update_data, **button_style)
button2.grid(row=1, column=0, padx=10, pady=10)

button3 = tk.Radiobutton(root, text="FF_Single_Neurons", variable=var, value='FF_Single_Neurons', command=update_data, **button_style)
button3.grid(row=2, column=0, padx=10, pady=10)

button4 = tk.Radiobutton(root, text="PC_PV_SST_Network", variable=var, value='PC_PV_SST_Network', command=update_data, **button_style)
button4.grid(row=3, column=0, padx=10, pady=10)

button5 = tk.Radiobutton(root, text="PC_PV_Network", variable=var, value='PC_PV_Network', command=update_data, **button_style)
button5.grid(row=4, column=0, padx=10, pady=10)

button6 = tk.Radiobutton(root, text="PC_Neuron", variable=var, value='PC_Neuron', command=update_data, **button_style)
button6.grid(row=5, column=0, padx=10, pady=10)

button7 = tk.Radiobutton(root, text="PV_Neuron", variable=var, value='PV_Neuron', command=update_data, **button_style)
button7.grid(row=6, column=0, padx=10, pady=10)

button8 = tk.Radiobutton(root, text="SST_Neuron", variable=var, value='SST_Neuron', command=update_data, **button_style)
button8.grid(row=7, column=0, padx=10, pady=10)

button9 = tk.Radiobutton(root, text="DE_PV_PC", variable=var, value='DE_PV_PC', command=update_data, **button_style)
button9.grid(row=8, column=0, padx=10, pady=10)

# Create the Run button
run_button = tk.Button(root, text="Run", command=start_dynapse_2, **button_style)
run_button.grid(row=9, column=0, padx=10, pady=20)

# Set the window size and position
root.geometry("250x450+400+400")

# Bind the window close event to the on_close function
root.protocol("WM_DELETE_WINDOW", on_close)

# Run the main event loop
root.mainloop()