from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset
df = pd.read_csv("dataset.csv")

# One-hot encoding untuk lokasi
df = pd.get_dummies(df, columns=["Lokasi"], drop_first=True)

# Pisahkan variabel independen (X) dan dependen (Y)
X = df.drop(columns=["Harga_Rumah"])
Y = df["Harga_Rumah"]

# Bagi dataset menjadi training dan testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Buat model regresi linear
model = LinearRegression()
model.fit(X_train, Y_train)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    plot_url = None
    prediction_table = None

    if request.method == "POST":
        luas_tanah = float(request.form["luas_tanah"])
        lokasi = request.form["lokasi"]

        # One-hot encoding untuk input user
        lokasi_encoded = [0, 0]  # Default: Bojongloa Kidul
        if lokasi == "Arcamanik":
            lokasi_encoded = [1, 0]
        elif lokasi == "Cibiru":
            lokasi_encoded = [0, 1]

        # Prediksi harga rumah
        input_data = np.array([luas_tanah] + lokasi_encoded).reshape(1, -1)
        prediction = model.predict(input_data)[0]

        # Visualisasi hasil prediksi
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=X_test["Luas_Tanah"], y=Y_test, color="blue", label="Data Aktual")
        sns.lineplot(x=X_test["Luas_Tanah"], y=model.predict(X_test), color="red", label="Prediksi")
        plt.xlabel("Luas Tanah (m2)")
        plt.ylabel("Harga Rumah (Miliar Rupiah)")
        plt.title("Prediksi Harga Rumah")
        plt.legend()

        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        # Buat tabel prediksi dengan styling yang lebih baik
        prediction_data = {
            'Luas Tanah (m²)': [f"{luas_tanah:,.0f}"],
            'Lokasi': [lokasi],
            'Prediksi Harga (Miliar)': [f"Rp {prediction:,.2f}"]
        }
        prediction_table = pd.DataFrame(prediction_data).to_html(
            classes='table table-bordered table-hover table-striped text-center',
            justify='center',
            index=False,
            escape=False
        )

    return render_template("index.html", prediction=prediction, plot_url=plot_url, prediction_table=prediction_table)

if __name__ == "__main__":
    app.run(debug=True)
