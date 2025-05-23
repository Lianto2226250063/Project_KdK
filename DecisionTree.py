import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.tree import DecisionTreeClassifier

# ===================== Load Dataset =====================
try:
    dataset = pd.read_csv("dataset.csv")
except FileNotFoundError:
    print("File dataset.csv tidak ditemukan! Pastikan file ada di direktori.")
    exit()

x = dataset.iloc[:, :-1]  # Fitur
y = dataset.iloc[:, -1]   # Label

# ===================== Training Model Decision Tree =====================
model = DecisionTreeClassifier()
model.fit(x, y)

# ===================== Fungsi Prediksi =====================
def predict():
    try:
        # Ambil input user dari GUI
        inputs = [float(entry.get()) for entry in entry_list]
        prediction = model.predict([inputs])
        if prediction[0] == 0:
            result = "Tidak Diabetes"
        else:
            result = "Diabetes"
        messagebox.showinfo("Hasil Prediksi", f"Prediksi hasil: {result}")
    except ValueError:
        messagebox.showerror("Input Error", "Semua input harus berupa angka!")

# ===================== GUI Tkinter =====================
window = tk.Tk()
window.title("Decision Tree Prediksi Diabetes")
window.geometry("400x500")

title_label = tk.Label(window, text="Prediksi Diabetes dengan Decision Tree", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10)

# Label fitur (8 input)
feature_names = dataset.columns[:-1]
entry_list = []

for feature in feature_names:
    lbl = tk.Label(window, text=f"{feature}:")
    lbl.pack()
    
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
