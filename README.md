# 📱 Ride-Hailing Apps Sentiment Analysis (Gojek, Grab, Maxim)

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.2-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.2.2-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

Repository ini berisi proyek **Analisis Sentimen** terhadap ulasan pengguna pada tiga aplikasi *ride-hailing* utama di Indonesia: **Gojek, Grab, dan Maxim**. Proyek ini merupakan bagian dari submission kelas *Machine Learning* di Dicoding.

Proyek ini mencakup seluruh *pipeline Machine Learning*, mulai dari *web scraping* data ulasan dari Google Play Store, pembersihan data, penanganan *class imbalance*, *preprocessing* teks bahasa Indonesia (menggunakan Sastrawi & NLTK), hingga pelatihan dan evaluasi beberapa model klasifikasi.

---

## 📊 Dataset Overview

Dataset yang digunakan dalam proyek ini dikumpulkan secara mandiri (*scraping*) dari Google Play Store.

* **Sumber Data:** Ulasan aplikasi `com.gojek.app`, `com.grabtaxi.passenger`, dan `com.taxsee.taxsee`.
* **Total Data Awal:** 9.402 ulasan (Gojek: 3.883 | Grab: 4.000 | Maxim: 4.000, setelah *cleaning* awal).
* **Distribusi Label Awal (Imbalanced):**
  * Positif: 5.342
  * Negatif: 3.616
  * Netral: 443
* **Data Setelah Balancing:** 16.026 ulasan (Distribusi rata: 33.3% untuk masing-masing kelas).
* **Data Final (Setelah Preprocessing):** 15.803 ulasan siap latih.

---

## ⚙️ Project Workflow

1. **Data Collection (`scraping.ipynb`):**
   Menggunakan *library* `google-play-scraper` untuk mengekstrak ribuan ulasan terbaru pengguna beserta rating dan *timestamp*-nya.
2. **Data Preprocessing & Balancing (`sentiment_analysis.ipynb`):**
   * Pembersihan teks (menghapus *punctuation*, angka, *link*).
   * *Oversampling/Undersampling* untuk menyeimbangkan kelas Positif, Negatif, dan Netral.
   * *Stopwords removal* menggunakan NLTK.
   * *Stemming* teks bahasa Indonesia menggunakan PySastrawi.
3. **Feature Extraction:**
   Menggunakan `TfidfVectorizer` untuk mengubah teks menjadi representasi numerik.
4. **Model Training & Evaluation:**
   Melatih 3 skema algoritma (*Support Vector Machine*, *Logistic Regression*, dan *Random Forest*) dengan pembagian rasio *train-test* yang berbeda untuk menemukan performa terbaik.

---

## 📈 Model Performance & Evaluation

Ketiga model dievaluasi untuk membandingkan tingkat akurasi pada fase *Training* dan *Testing*. Berikut adalah ringkasan hasilnya:

| Skema Algoritma | Split Ratio (Train/Test) | Akurasi Training | Akurasi Testing | Status |
| :--- | :---: | :---: | :---: | :---: |
| **SVM + TF-IDF** | 80/20 | 98.16% | 95.03% | ✅ LULUS |
| **Logistic Reg. + TF-IDF** | 80/20 | 97.31% | 94.08% | ✅ LULUS |
| **Random Forest + TF-IDF** | **70/30** | **99.26%** | **96.06%** | **🌟 TERBAIK** |

Model **Random Forest** terpilih sebagai model terbaik dengan *F1-Score* yang sangat stabil di angka **0.96** untuk makro dan rata-rata tertimbang, sehingga model ini diekspor untuk *inference*.

---

## 📂 Repository Structure

```text
📁 sentiment-analysis-dicoding
├── scraping.ipynb             # Notebook untuk scraping ulasan dari Google Play
├── sentiment_analysis.ipynb   # Notebook utama (Preprocessing, EDA, Training, Evaluasi)
├── requirements.txt           # Daftar dependensi library
├── dataset_gabungan.csv       # Dataset mentah hasil scraping gabungan 3 aplikasi
├── dataset_final.csv          # Dataset setelah class balancing
├── dataset_preprocessed.csv   # Dataset final setelah cleansing, stemming, dll
├── best_model.pkl             # Model Random Forest terbaik yang telah dilatih
├── best_tfidf.pkl             # TF-IDF Vectorizer yang telah di-fit pada data latih
└── label_encoder.pkl          # Encoder untuk mapping label (Negatif, Netral, Positif)
```

---

## 🚀 How to Run (Local Installation)

Jika Anda ingin menjalankan proyek ini di komputer lokal Anda:

1. **Clone repository ini:**
   ```bash
   git clone [https://github.com/USERNAME_GITHUB_KAMU/ride-hailing-sentiment-analysis.git](https://github.com/USERNAME_GITHUB_KAMU/ride-hailing-sentiment-analysis.git)
   cd ride-hailing-sentiment-analysis
   ```

2. **Buat Virtual Environment (Opsional namun disarankan):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Linux/Mac
   venv\Scripts\activate     # Untuk Windows
   ```

3. **Install dependensi yang dibutuhkan:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Buka file `sentiment_analysis.ipynb` atau uji model secara manual menggunakan file `.pkl` yang tersedia.

---

## 💡 Contoh Inference Manual

Anda dapat memuat model `.pkl` dan melakukan prediksi pada teks baru:

```python
import joblib

# Load Model dan Vectorizer
model = joblib.load('best_model.pkl')
tfidf = joblib.load('best_tfidf.pkl')

# Teks Ulasan Baru
teks_baru = ["Aplikasi bagus sekali, driver cepat sampai"]
teks_vektor = tfidf.transform(teks_baru)

# Prediksi
prediksi = model.predict(teks_vektor)
print("Hasil Sentimen:", prediksi)
# Output: 😊 POSITIF (Confidence: 100.0%)
```

---

**Author:** Destian Aldi Nugraha  
*Informatics Student | Enthusiast in Machine Learning & Data Science*
