# Analisis Sentimen Ulasan Aplikasi Ride-Hailing (Gojek, Grab, Maxim)

**Summary (English).** Sentiment classification of Indonesian Google Play reviews for Gojek, Grab and Maxim: scraping, Indonesian text preprocessing (Sastrawi, NLTK), TF-IDF, and three classifiers (linear SVM, Logistic Regression, Random Forest). Labels are derived from star ratings, not hand-annotated. On a leakage-free evaluation the models reach 83-85% test accuracy against a 56% majority-class baseline, and they almost never recover the neutral class (recall 2-15%). An earlier version of this README reported 96%; that figure came from oversampling the whole dataset before the train/test split, so most test reviews also appeared in training. See "Catatan koreksi" below.

Proyek ini bagian dari submission kelas *Machine Learning* di Dicoding. Isinya seluruh alur: *scraping* ulasan dari Google Play Store, pembersihan data, *preprocessing* teks bahasa Indonesia, penanganan ketidakseimbangan kelas, lalu pelatihan dan evaluasi tiga model.

## Catatan koreksi (5 Nov 2026)

Versi pertama repositori ini melaporkan akurasi uji 95-96%. Angka itu **tidak valid**. Penyebabnya ada di `sentiment_analysis.ipynb`: *oversampling* acak dengan duplikasi (`resample(..., replace=True)`) dijalankan pada **seluruh** dataset, baru setelah itu `train_test_split`. Akibatnya salinan ulasan yang sama muncul di data latih dan data uji.

Saya memeriksanya dengan mereplikasi metode lama: **79-82% baris uji punya teks bersih yang identik dengan teks di data latih**. Setelah split dilakukan lebih dulu, akurasi uji turun ke 83-85%.

Notebook asli sengaja tidak diubah, supaya jejaknya terlihat. Evaluasi yang benar ada di `evaluate_no_leakage.py`, dan keluarannya di `results_no_leakage.json`. Beberapa angka lain di README lama juga keliru dan sudah diperbaiki di bawah.

## Dataset

* **Sumber:** ulasan aplikasi `com.gojek.app`, `com.grabtaxi.passenger`, dan `com.taxsee.taxsee` dari Google Play Store, di-*scrape* sendiri.
* **Hasil scraping (`dataset_gabungan.csv`):** 9.401 ulasan. Gojek 3.883, Grab 3.051, Maxim 2.467. (README lama menulis Grab 4.000 dan Maxim 4.000; itu salah.)
* **Label diturunkan dari rating bintang, bukan dianotasi manual:** rating 1-2 = negatif, 3 = netral, 4-5 = positif. Jadi model memprediksi kelompok rating dari teks, bukan sentimen hasil anotasi manusia.
* **Distribusi label:** positif 5.342, negatif 3.616, netral 443.
* **Setelah pembersihan** (ulasan yang kosong setelah *preprocessing* dibuang): **9.227 ulasan unik**, yaitu positif 5.191 (56,3%), negatif 3.598 (39,0%), netral 438 (4,7%). Tersimpan di `dataset_unique_clean.csv` tanpa kolom username.
* `dataset_final.csv` (16.476 baris) dan `dataset_preprocessed.csv` (16.263 baris) adalah hasil *oversampling* seluruh data pada versi lama. Keduanya hanya memuat 6.365 dari ulasan asli yang unik. Disimpan untuk transparansi; jangan dipakai untuk evaluasi.

## Alur kerja

1. **Pengumpulan data (`scraping.ipynb`):** `google-play-scraper` untuk ulasan, rating, dan waktu.
2. **Preprocessing (`sentiment_analysis.ipynb`):** huruf kecil, hapus URL, angka, dan tanda baca; normalisasi slang; *stopwords* NLTK ditambah daftar kustom; *stemming* dengan PySastrawi.
3. **Ekstraksi fitur:** `TfidfVectorizer` (20.000 fitur, unigram dan bigram, `sublinear_tf`, `min_df=2`, `max_df=0.95`), di-*fit* hanya pada data latih.
4. **Pelatihan dan evaluasi (`evaluate_no_leakage.py`):** split *stratified* dulu, *oversampling* hanya pada data latih, uji pada data uji dengan distribusi alami.

## Hasil (evaluasi tanpa kebocoran)

`random_state=42`, scikit-learn 1.8.0. Data uji memakai distribusi kelas alami. Penebak kelas mayoritas ("selalu positif") mencapai akurasi **56,3%**.

| Model + TF-IDF | Split | Akurasi latih | Akurasi uji | F1 makro | Recall negatif / netral / positif |
| :--- | :---: | :---: | :---: | :---: | :---: |
| SVM linear | 80/20 | 98,1% | **84,9%** | 0,607 | 0,889 / 0,057 / 0,889 |
| Logistic Regression | 80/20 | 97,5% | 83,6% | **0,625** | 0,858 / 0,148 / 0,879 |
| Random Forest (300 pohon) | 70/30 | 99,3% | 83,2% | 0,579 | 0,846 / 0,023 / 0,891 |

Uji yang lebih ketat, dengan membuang baris uji yang teks bersihnya juga ada di data latih (sekitar 20% baris uji): akurasi 82,5% (SVM), 80,8% (LR), 80,6% (RF).

Yang terbaca dari hasil ini:

* Model cukup baik membedakan negatif dari positif (recall sekitar 0,85-0,89), tetapi **hampir tidak pernah menebak netral** (recall 0,02-0,15). Kelas netral hanya rating 3 dengan 438 ulasan unik, dan teks ulasan bintang 3 mungkin sulit dibedakan dari ulasan negatif atau positif (dugaan; belum diperiksa per contoh). *Oversampling* hanya menduplikasi contoh yang sedikit itu.
* Selisih antar model kecil dan memakai satu kali split, jadi tidak cukup bukti untuk menyebut satu model terbaik. SVM unggul di akurasi, Logistic Regression di F1 makro.
* Akurasi tinggi pada data tak seimbang menyesatkan. F1 makro dan recall per kelas lebih jujur.

Hasil metode lama, hanya sebagai pembanding (dengan kebocoran): SVM 94,4%, LR 93,8%, RF 95,7% pada replikasi saya. Versi pertama README menulis 95,03%, 94,08%, dan 96,06%.

## Batasan

* Label berasal dari rating, dan rating tidak selalu mencerminkan isi teks, sehingga ada derau pada label.
* Satu split per skema, tanpa validasi silang.
* Kelas netral sangat sedikit. Pembobotan kelas, penyesuaian ambang, data netral tambahan, atau model berbasis IndoBERT adalah langkah lanjutan yang masuk akal.
* Ulasan bertanggal 9 Maret sampai 13 April 2026 (satu jendela scraping sekitar lima minggu), sehingga belum tentu mewakili periode lain.

## Struktur repositori

```text
.
├── scraping.ipynb               # scraping ulasan dari Google Play
├── sentiment_analysis.ipynb     # notebook asli (preprocessing, balancing, pelatihan); evaluasinya mengandung kebocoran
├── evaluate_no_leakage.py       # evaluasi yang benar + pembanding metode lama
├── results_no_leakage.json      # keluaran evaluate_no_leakage.py
├── dataset_gabungan.csv         # data mentah hasil scraping (9.401 ulasan)
├── dataset_unique_clean.csv     # 9.227 ulasan unik yang sudah dibersihkan, tanpa username
├── dataset_final.csv            # hasil oversampling seluruh data (versi lama)
├── dataset_preprocessed.csv     # hasil preprocessing data hasil oversampling (versi lama)
├── best_tfidf.pkl               # TF-IDF dari submission asli
├── label_encoder.pkl            # pemetaan label
└── requirements.txt
```

`best_model.pkl` tidak disertakan di repositori. `best_tfidf.pkl` berasal dari submission asli, jadi jangan dianggap bagian dari evaluasi yang sudah dikoreksi.

## Cara menjalankan

```bash
git clone https://github.com/DeAlNu/Indonesian-ride-hailing-reviews.git
cd Indonesian-ride-hailing-reviews
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python evaluate_no_leakage.py
```

Skrip tidak memerlukan Sastrawi karena teks bersih sudah ada di `dataset_unique_clean.csv`. Untuk mengulang *preprocessing* dari ulasan mentah, jalankan sel *preprocessing* di `sentiment_analysis.ipynb` pada `dataset_gabungan.csv`.

---

**Author:** Destian Aldi Nugraha. Informatics graduate, Universitas Ahmad Dahlan. [Portfolio](https://dealnu.vercel.app)
