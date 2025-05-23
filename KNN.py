import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split
import math
import numpy as np

# ===================== Fungsi Manhattan Distance =====================
def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))

# ===================== Fungsi KNN =====================
def knn_classify(x_train, y_train, x_test, k):
    predictions = [] 

    for test_point in x_test:
        distances = []  

        for i, train_point in enumerate(x_train.values):
            distance = manhattan_distance(train_point, test_point)
            distances.append((distance, y_train.values[i]))

        distances.sort(key=lambda x: x[0])
        k_nearest_neighbors = distances[:k]
        
        label_counts = {}
        for neighbor in k_nearest_neighbors:
            label = neighbor[1]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        predicted_label = max(label_counts, key=label_counts.get)
        predictions.append(predicted_label)
    
    return predictions

# ===================== Load Dataset =====================
try:
    dataset = pd.read_csv("dataset.csv")
except FileNotFoundError:
    print("File dataset.csv tidak ditemukan! Pastikan file ada di direktori.")
    exit()

x = dataset.iloc[:, :-1]  
y = dataset.iloc[:, -1]   

# Parameter K
k = 11

# ===================== Fungsi Prediksi =====================
def predict():
    try:
        # Ambil input user dari GUI
        inputs = [float(entry.get()) for entry in entry_list]
        prediction = knn_classify(x, y, [inputs], k)
        if prediction[0] == 0:
            result = "Tidak Diabetes"
        else:
            result = "Diabetes"
        messagebox.showinfo("Hasil Prediksi", f"Prediksi hasil: {result}")
    except ValueError:
        messagebox.showerror("Input Error", "Semua input harus berupa angka!")

# ===================== GUI Tkinter =====================
window = tk.Tk()
window.title("KNN Prediksi Diabetes")
window.geometry("400x500")

title_label = tk.Label(window, text="Prediksi Diabetes dengan KNN", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10)

# Label fitur (8 input)
feature_names = dataset.columns[:-1]
entry_list = []

for feature in feature_names:
    # Label untuk nama fitur
    lbl = tk.Label(window, text=f"{feature}:")
    lbl.pack()
    
    # Entry box untuk input user
    entry = tk.Entry(window)
    entry.pack()
    entry_list.append(entry)

# Tombol Prediksi
predict_button = tk.Button(window, text="Prediksi", command=predict, bg="green", fg="white")
predict_button.pack(pady=20)

# Tombol Keluar
exit_button = tk.Button(window, text="Keluar", command=window.destroy, bg="red", fg="white")
exit_button.pack(pady=10)

# Jalankan GUI
window.mainloop()
