import pickle
import joblib
import pandas as pd
import numpy as np
from keras import Model
import tensorflow as tf

import joblib
import tkinter as tk

cols = ['SELLINGMARK', 'GRADE', 'BUYER', 'Valuation', 'UpperValuation', 'LEAFStandard', 'INFUSEDRemarks',
        'LiquorRemark', 'MinPrice', 'AskingPrice']

# model = joblib.load('randomSearchML1.pkl')
model = joblib.load('xgb_modelHP.pkl')

# model = pickle.load(open('RF2model.pkl', 'rb'))

# df = pd.DataFrame(columns=cols)
# df['SELLINGMARK'] = [input("Please input Selling Mark: ")]
# df['GRADE'] = [input("Please input GRADE: ")]
# df['BUYER'] = [input("Please input BUYER: ")]
# df['Valuation'] = [float(input("Please input Valuation: "))]
# df['UpperValuation'] = [float(input("Please input UpperValuation: "))]
# df['LEAFStandard'] = [input("Please input LEAFStandard: ")]
# df['INFUSEDRemarks'] = [input("Please input INFUSEDRemarks: ")]
# df['LiquorRemark'] = [input("Please input LiquorRemark: ")]
# df['MinPrice'] = [float(input("Please input MinPrice: "))]
# df['AskingPrice'] = [float(input("Please input AskingPrice: "))]

root = tk.Tk()


GRADE_label = tk.Label(root, text='GRADE:')
GRADE_label.pack()
GRADE_entry = tk.Entry(root)
GRADE_entry.pack()

BUYER_label = tk.Label(root, text='BUYER:')
BUYER_label.pack()
BUYER_entry = tk.Entry(root)
BUYER_entry.pack()

Valuation_label = tk.Label(root, text='Valuation:')
Valuation_label.pack()
Valuation_entry = tk.Entry(root)
Valuation_entry.pack()

UpperValuation_label = tk.Label(root, text='UpperValuation:')
UpperValuation_label.pack()
UpperValuation_entry = tk.Entry(root)
UpperValuation_entry.pack()

MinPrice_label = tk.Label(root, text='MinPrice:')
MinPrice_label.pack()
MinPrice_entry = tk.Entry(root)
MinPrice_entry.pack()

AskingPrice_label = tk.Label(root, text='AskingPrice:')
AskingPrice_label.pack()
AskingPrice_entry = tk.Entry(root)
AskingPrice_entry.pack()

Bold_label = tk.Label(root, text='Bold:')
Bold_label.pack()
Bold_entry = tk.Entry(root)
Bold_entry.pack()

Clean_label = tk.Label(root, text='Clean:')
Clean_label.pack()
Clean_entry = tk.Entry(root)
Clean_entry.pack()

Dark_label = tk.Label(root, text='Dark:')
Dark_label.pack()
Dark_entry = tk.Entry(root)
Dark_entry.pack()

Mixed_label = tk.Label(root, text='Mixed:')
Mixed_label.pack()
Mixed_entry = tk.Entry(root)
Mixed_entry.pack()

Plain_label = tk.Label(root, text='Plain:')
Plain_label.pack()
Plain_entry = tk.Entry(root)
Plain_entry.pack()

Quality_label = tk.Label(root, text='Quality:')
Quality_label.pack()
Quality_entry = tk.Entry(root)
Quality_entry.pack()


# Define a function to make a prediction when the user clicks the button
def predict_price():
    # Get the house size from the text box

    GRADE = int(GRADE_entry.get())
    BUYER = int(BUYER_entry.get())
    Valuation = float(Valuation_entry.get())
    UpperValuation = float(UpperValuation_entry.get())
    MinPrice = float(MinPrice_entry.get())
    AskingPrice = float(AskingPrice_entry.get())
    Bold = int(Bold_entry.get())
    Clean = int(Clean_entry.get())
    Dark = int(Dark_entry.get())
    Mixed = int(Mixed_entry.get())
    Plain = int(Plain_entry.get())
    Quality = int(Quality_entry.get())

    # Use the trained model to make a prediction
    price = model.predict([[GRADE, BUYER, Valuation, UpperValuation, MinPrice, AskingPrice, Bold, Clean, Dark, Mixed, Plain, Quality ]])


    # Display the predicted price in a label
    result_label.configure(text=f'Predicted price: Rs{price[0]:,.2f}')


# Add a button to trigger the prediction
predict_button = tk.Button(root, text='Predict price', command=predict_price)
predict_button.pack()

# Add a label to display the predicted price
result_label = tk.Label(root, text='')
result_label.pack()

# Run the UI
root.mainloop()

# pred = model.predict(df)
#
# print(pred)
