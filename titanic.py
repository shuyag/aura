from __future__ import print_function

import numpy as np
import tflearn

import tkinter


def close_window(): 
    window.destroy()

def close_results(): 
    results.destroy()
    
window = tkinter.Tk()

vname1 = tkinter.StringVar()
vsex1 = tkinter.StringVar()
vage1 = tkinter.IntVar()
vcls1 = tkinter.IntVar()
vhrs1 = tkinter.IntVar()

vname2 = tkinter.StringVar()
vsex2 = tkinter.StringVar()
vage2 = tkinter.IntVar()
vcls2 = tkinter.IntVar()
vhrs2 = tkinter.IntVar()

window.title("Aura")
window.geometry("250x150+300+300")
window.wm_iconbitmap('aura.ico')
window.configure(background="#a1dbcd")

image1 = tkinter.PhotoImage(file="hero.gif")
w = image1.width()
h = image1.height()
window.geometry("%dx%d+0+0" % (w, h))
panel1 = tkinter.Label(window, image=image1)
panel1.grid(row = 0, column = 0)
panel1.image = image1

#lbl1 = tkinter.Label(window, text="User 1", fg="#383a39", bg="#a1dbcd", font=(" Helvetica", 16))
enta1 = tkinter.Entry(window, text= "Name", textvariable = vname1)
enta2 = tkinter.Entry(window,textvariable = vage1)
enta3 = tkinter.Entry(window,textvariable = vsex1)
cls1 = tkinter.Scale(window, from_=1, to=3, variable = vcls1, orient=tkinter.HORIZONTAL)
enta4 = tkinter.Entry(window,textvariable = vhrs1)

#lbl2= tkinter.Label(window, text="User 2", fg="#383a39", bg="#a1dbcd", font=(" Helvetica", 16))
entb1 = tkinter.Entry(window,textvariable = vname2)
entb2 = tkinter.Entry(window,textvariable = vage2)
entb3 = tkinter.Entry(window,textvariable = vsex2)
cls2 = tkinter.Scale(window, from_=1, to=3, variable = vcls2, orient=tkinter.HORIZONTAL)
entb4 = tkinter.Entry(window,textvariable = vhrs2)

btn = tkinter.Button(window, text="Submit", command=close_window)

#lbl1.place(x=300,y=100)
enta1.place(x=100,y=280)
enta2.place(x=100,y=330)
enta3.place(x=100,y=380)
cls1.place(x=100,y=430)
enta4.place(x=100,y=500)

#lbl2.place(x=700,y=100)
entb1.place(x=730,y=280)
entb2.place(x=730,y=330)
entb3.place(x=730,y=380)
cls2.place(x=730,y=430)
entb4.place(x=730,y=500)

btn.place(x=450, y=550, anchor=tkinter.CENTER)


window.mainloop()

name1 = vname1
class1 = 4-vcls1.get()
sex1 = vsex1
age1 = vage1.get()
tix1 = vhrs1.get()*20

name2 = vname2
class2 = 4-vcls2.get()
sex2 = vsex2
age2 = vage2.get()
tix2 = vhrs2.get()*20

# Download the Titanic dataset
#from tflearn.datasets import titanic
#titanic.download_dataset('titanic_dataset.csv')

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
data, labels = load_csv('titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)


# Preprocessing function
def preprocess(data, columns_to_ignore):
    # Sort by descending id and delete columns
    for id in sorted(columns_to_ignore, reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
      # Converting 'sex' field to float (id is 1 after removing labels column)
      data[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(data, dtype=np.float32)

# Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
to_ignore=[1, 6]

# Preprocess data
data = preprocess(data, to_ignore)

# Build neural network
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)


# Let's create some data for DiCaprio and Winslet
dicaprio = [class1, name1, sex1, age1, 0, 0, 'N/A', tix1]
winslet = [class2, name2, sex2, age2, 1, 2, 'N/A', tix2]
# Preprocess data
dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)
# Predict surviving chances (class 1 results)
pred = model.predict([dicaprio, winslet])
print("Percent Chance User A Agrees:", pred[0][1])
print("Percent Chance User B Agrees:", pred[1][1])

preda = pred[0][1]
predb = pred[1][1]

rest1 = str(preda)
rest2 = str(predb)

stk1 = "Percent Chance User A Agrees: "
stk2 = "Percent Chance User B Agrees: "

str1 = stk1 + rest1
str2 = stk2 + rest2

results = tkinter.Tk()
results.title("Aura")
results.geometry("250x150+300+300")
results.wm_iconbitmap('aura.ico')
results.configure(background="#a1dbcd")

image2 = tkinter.PhotoImage(file="hero1.gif")
w1 = image2.width()
h1 = image2.height()

results.geometry("%dx%d+0+0" % (w1, h1))
panel2 = tkinter.Label(results, image=image2)
panel2.grid(row = 0, column = 0)
panel2.image = image2

res1= tkinter.Label(results, text= rest1, fg="#000000", font=(" Helvetica", 24))
res2= tkinter.Label(results, text= rest2, fg="#000000", font=(" Helvetica", 24))
btngo = tkinter.Button(results, text="Close", command=close_results)

res1.place(x=50,y=480)
res2.place(x=600,y=480)
btngo.place(x=450, y=550)

results.mainloop()
