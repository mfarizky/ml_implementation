# Laporan Proyek Machine Learning - Rijal Muhammad Farizky
![bbca](.\img\bbca.png)
## Domain Proyek
Domain yang dipilih untuk proyek machine learning ini adalah **Keuangan**, dengan judul **Predictive Analytics: BBCA Stock Price Forecasting**

## Business Understanding
Pasar saham merupakan sistem yang kompleks dan dinamis, di mana harga saham dipengaruhi oleh berbagai faktor seperti indikator ekonomi, kinerja perusahaan, sentimen pasar, dan peristiwa geopolitik [1]. Peramalan harga saham yang akurat sangat penting bagi investor, analis keuangan, dan pembuat kebijakan untuk pengambilan keputusan yang tepat, optimalisasi portofolio, dan mitigasi risiko [2]. Analisis dan Peramalan data *time series* menjadi salah satu alat penting dalam menganalisis data historis harga saham dan memprediksi pergerakan di masa depan [3].

Menurut data dari [id.investing.com](https://id.investing.com/equities/most-active-stocks), saham BBCA (Bank Central Asia) merupakan salah satu saham yang paling aktif diperdagangkan di Indonesia. BBCA adalah kode saham untuk PT. Bank Central Asia Tbk., sebuah bank yang beroperasi di sektor keuangan. Bank ini menjadi salah satu pemain utama di sektor keuangan Indonesia. Pada bulan **Januari 2024**, nilai kapitalisasi pasar BCA tercatat sebesar **Rp1.165 Triliun**. Proporsi nilai yang setara 10,21% dari total kapitalisasi pasar IDX [[4]](https://dataindonesia.id/pasar-saham/detail/data-8-saham-dengan-kapitalisasi-pasar-terbesar-big-caps-di-bei-per-januari-2024). Hal ini yang menunjukkan betapa penting dan berpengaruhnya saham BBCA di pasar saham.

### Problem Statements
1. Bagaimana cara membuat model prediksi harga saham BBCA dengan akurat dengan menggunakan data time series ?
2. Bagaimana cara menentukan model terbaik untuk prediksi harga saham BBCA?

### Goals
1. Membuat model deep learning yang dapat memprediksi harga saham BBCA berdasarkan data time series.
2. Membangun dan membandingkan 2 algoritma *deep learning* untuk mendapatkan model terbaik untuk prediksi harga saham BBCA.


### Solution Statements
1. Mengembangkan model yang dapat memprediksi harga saham BBCA yang merupakan data *time series*. Dalam hal ini menggunakan pendekatan deep learning dengan algoritma LSTM. **LSTM lebih unggul** dibandingkan algoritma deep learning lainnya seperti RNN (Recurrent Neural Network) dan CNN (Convolutional Neural Network) [[5]](https://www.researchgate.net/publication/355804252_A_Comparative_Study_of_Stock_Forecasts_by_LSTM_and_RNN_Neural_Networks). Salah satu kelebihan LSTM adalah kemampuannya mengatasi masalah vanishing gradient. *vanishing gradient* terjadi ketika gradien menjadi sangat kecil saat backpropagation, sehingga jaringan sulit untuk mempelajari pola jangka panjang. Kelebihan ini membuat LSTM mampu mempertahankan informasi temporal yang memungkinkannya untuk mempelajari pola tren harga saham secara lebih akurat [[6]](https://www.researchgate.net/publication/379811995_Stock_Market_Analysis_and_Prediction_Using_LSTM_A_Case_Study_on_Technology_Stocks).

2. Salah satu kelebihan utama dari LSTM adalah kemampuannya mengatasi masalah vanishing gradients, yang sering menjadi kendala pada jaringan saraf tradisional saat mempelajari data sequence panjang. Sebagai alternatif, algoritma **GRU (Gated Recurrent Unit)** juga memiliki keunggulan serupa, namun dengan arsitektur yang lebih sederhana dan efisien secara komputasi karena menggunakan lebih sedikit parameter. Meski GRU lebih cepat dalam proses pelatihan, pendekatan utama dalam kasus ini tetap berfokus pada akurasi prediksi. Untuk memastikan model terbaik dalam memprediksi harga saham BBCA, dilakukan perbandingan kinerja antara LSTM dan GRU. Keduanya dievaluasi menggunakan berbagai metrik, seperti **RMSE (Root Mean Square Error)**, **MAE (Mean Absolute Error)**, **MAPE (Mean Absolute Percentage Error)**, dan **RMSPE (Root Mean Square Percentage Error)**.

## Data Understanding
### Data Description
Data Bank Central Asia [(BBCA)](https://www.kaggle.com/datasets/caesarmario/bank-central-asia-stock-historical-price) Stock Historical Price berisi informasi harga saham BBCA (Bank Central Asia) sejumlah 1498 data *time series* mulai dari tanggal 1 Januari 2019 hingga 14 Februari 2025. Data ini mencakup 7 variabel yang memberikan gambaran lengkap tentang aktivitas perdagangan saham BBCA selama periode tersebut.

Berikut adalah penjelasan dari masing-masing variabel dalam data historis harga saham BBCA:
* **Date** : Tanggal harga saham dicatat. [Object/String]
* **Open (Harga Pembukaan):** Harga saham BBCA pada awal hari perdagangan. Ini adalah harga yang disepakati pada saat pasar dibuka. [Float]
* **High (Harga Tertinggi):** Harga saham BBCA tertinggi yang tercapai selama hari perdagangan. [Float]
* **Low (Harga Terendah):** Harga saham BBCA terendah yang tercapai selama hari perdagangan. [Float]
* **Close (Harga Penutupan):** Harga saham BBCA pada akhir hari perdagangan. Ini adalah harga yang disepakati pada saat pasar ditutup. [Float]
* **Adj Close (Harga Penutupan yang Disesuaikan)**: Harga penutupan saham yang telah disesuaikan untuk memperhitungkan Corporate Action  seperti dividen, stock split, dan lainnya. Tujuannya adalah untuk memberikan gambaran yang lebih akurat tentang kinerja saham dari waktu ke waktu. [Float]
* **Volume:** Jumlah saham BBCA yang diperdagangkan selama hari perdagangan. Volume yang tinggi menunjukkan minat yang besar terhadap saham tersebut begitu pula sebaliknya. [Integer]

### Exploratory Data Analysis
**Open and Close Prices of All Time**

![](.\img\open-close-plot2.png)

Berdasarkan data tersebut, dapat diamati bahwa harga saham BBCA cenderung mengalami kenaikan tiap tahunnya dan membentuk pola tertentu. Pola ini nantinya dapat dipelajari oleh model deep learning untuk membuat prediksi yang lebih akurat di masa depan.

**Volume of All time**

![](.\img\volume-ofAllTime2.png) 

Volume saham yang diperdagangkan cenderung konsisten tiap tahunnya mengindikasikan bahwa saham tersebut memiliki likuiditas yang stabil. Likuiditas yang konsisten menandakan bahwa saham tersebut dapat diperdagangkan dengan mudah oleh investor tanpa menyebabkan perubahan harga yang signifikan, yang merupakan tanda pasar yang sehat dan aktif, meskipun volume saham melonjak cukup tinggi pada periode tertentu.

**Close Yearly Average Movement**

**Close** merupakan harga penutupan saham pada akhir hari perdagangan. Harga ini sering digunakan sebagai acuan dalam analisis teknis dan keputusan trading karena mencerminkan nilai terakhir dari saham pada hari tersebut, yang biasanya dianggap lebih penting daripada harga lainnya dalam sehari. Kali ini kita akan melihat pergerakan rata-rata harga penutupan (Close) tiap tahunnya.

![](.\img\Close-YearlyAverageMovement2.png)

Berdasarkan plot di atas, dapat dilihat bahwa pergerakan saham cenderung memiliki pola tertentu dan cukup konsisten tiap tahunnya. Meskipun terdapat fluktuasi bulanan, saham cenderung menunjukkan tren kenaikan dari tahun ke tahun.

**Feature Correlation**

Selanjutnya, kita akan menganalisi `Numerical Feature Correlation`untuk mengetahui hubungan antara fitur-fitur numerik dalam data. Kita akan melakukan visualisasi hubungan antar fitur numerik dengan pair plot kemudian menghitung korelasi matriks untuk mendapatkan gambaran lengkap tentang hubungan antar semua fitur numerik.
![](.\img\pair_plot2.png)

Visualisasi di atas menunjukkan terdapat **korelasi** antar fitur `Open`, `High`, `Low`, `Close`, `Adj. Close`. Selanjutnya menghitung korelasi matrix.

![](.\img\corr_matrix2.png)

Sama seperti visualisasi pair-plot, korelasi antar fitur `Open`, `High`, `Low`, `Close`, `Adj. Close` **bernilai 1** yang menunjukkan korelasi **positif sempurna**.

## Data Preparation
Berikut merupakan tahapan-tahapan dalam Data Preparation:

### Feature Selection
Berdasarkan hasil analisis, fitur `Open`, `High`, `Low`, `Close`, `Adj. Close` berkorelasi positif sempurna yang berarti satu perubahan akan saling memengaruhi, maka kita bisa memilih salah satu fitur tersebut untuk digunakan dalam membangun Model Deep Learning. Dalam hal ini, kita akan menggunakan fitur `Close`.

### Normalisasi data dengan StandardScaler
Data diubah ke dalam **skala standar** (**rata-rata = 0**, **standar deviasi = 1**). Normalisasi ini penting agar model machine learning, khususnya yang berbasis gradient descent, dapat bekerja lebih optimal tanpa bias terhadap skala fitur yang berbeda.

### Splitting data menjadi train dan test dengan rasio 90:10
Data dibagi menjadi **90% untuk pelatihan (train)** dan **10% untuk pengujian (test)**. Rasio ini memastikan model memiliki cukup data untuk belajar sambil tetap menyediakan data pengujian untuk mengevaluasi kinerjanya pada data baru yang belum pernah dilihat sebelumnya.
![Splitdata](.\img\split_data.png)
### Membuat window
Windowing adalah proses membagi data urutan waktu (time series) menjadi beberapa subset data atau `window` dengan ukuran tertentu (`window size`). Dalam kasus ini kita membuat window dengan window size sebesar **60 hari**. Hasil proses ini sebagai berikut.

![Hasil Preparation](.\img\preparation_result.png)

## Modeling
Pada tahap ini, kita akan membangun model **LSTM** dan **GRU** dengan parameter yang sama. Berikut adalah konfigurasi model yang akan kita gunakan:
| Parameter                      | Value      |
|--------------------------------|------------|
| **EPOCHS**                     | 20         |
| **Hidden Layer**               | 128 units  |
| **Activation Function**        | Adam       |
| **Look Back Value / Window**   | 60         |
| **Dropout**                    | 0.2        |
| **Loss Function**              | MAE        |
| **Metric Function**            | RMSE       |
### LSTM Model

Berikut merupakan arsitektur model LSTM.

![LSTM_MODEL](.\img\LSTM_Model.png)

Seperti yang sudah dijelaskan pada  problem statement bahwa **LSTM**  memiliki kemampuan vanishing gradients yang membuatnya mampu
mempertahankan informasi temporal yang memungkinkannya untuk mempelajari pola tren harga saham secara lebih akurat.

### GRU Model
Berikut merupakan arsitektur model GRU

![GRU_MODEL](.\img\GRU_Model.png)

Dengan kelebihan serupa dengan LSTM, **GRU** hadir dengan arsitektur cenderung lebih sederhana, akan tetapi pada kasus ini kita memntukan model terbaik berdasarkan hasil metrik evaluasi.

## Evaluation

### Perbandingan Train Loss and Validation Loss

![](.\img\EVAL-loss.png)
- **Grafik LSTM** menunjukkan bahwa train loss cenderung stabil dan memiliki nilai yang relatif rendah seiring bertambahnya epoch. Namun, validation loss menunjukkan fluktuasi yang signifikan dan tidak stabil. Hal ini bisa mengindikasikan adanya overfitting, di mana model bekerja dengan baik pada data training tetapi kurang baik pada data validasi.
- **Pada grafik GRU**, train loss juga menurun dan stabil di nilai rendah seiring bertambahnya epoch.
Sama seperti LSTM, validation loss pada GRU juga berfluktuasi tetapi dengan pola yang lebih terlihat dibandingkan LSTM. Fluktuasi ini mungkin menandakan bahwa model belum berhasil menangkap pola secara konsisten pada data validasi, meskipun overfitting mungkin lebih sedikit dibandingkan LSTM.

### Perbandingan Hasil Prediksi vs Data Tes

![](.\img\EVAL_PredictionResult.png)
- Prediksi menggunakan **LSTM** `garis oranye` memiliki pola yang cukup mirip dengan data aktual pada test set `garis hijau`. Hal ini menunjukkan bahwa LSTM dapat menangkap tren data dengan baik. Pada beberapa titik, prediksi sedikit melenceng dari data aktual, terutama saat terjadi perubahan tajam (spike). Prediksi LSTM juga cenderung berada di bawah garis hijau (data aktual), berarti model LSTM memiliki kecenderungan underestimation pada beberapa titik. Namun secara keseluruhan, model mampu mempertahankan akurasi yang cukup baik untuk data test.
- Prediksi menggunakan **GRU** `garis oranye` juga hampir serupa dengan LSTM. Perbedaannya hanya terletak pada kecenderungannya berada di atas garis hijau (data aktual) yang berarti model GRU memiliki kecenderungan overestimation. Secara keseluruhan, model mampu mempertahankan akurasi yang cukup baik untuk data test.

### Perbandingan Metrik Evaluasi
Pada tahap ini kita membandingkan LSTM  dan GRU dengan metrik:
| **Metrik** | **Deskripsi**                                                                                                                                         |
|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| **MAE**    | Untuk menunjukkan seberapa besar deviasi prediksi dari data sebenarnya secara umum.       |
| **RMSE**   | Untuk mendeteksi kesalahan yang lebih besar.                                              |
| **MAPE**   | Untuk memudahkan keterbacaan MAE karena menyajikan bobot dalam bentuk persentase.         |
| **RSMPE**  | Sama halnya dengan MAPE, tetapi untuk mendeteksi kesalahan yang lebih besar               |


![](.\img\EVAL_METRIC.png)

**Deskripsi Hasil:**
* MAE: LSTM (114.60) lebih rendah daripada GRU (152.32), menunjukkan prediksi LSTM lebih akurat secara rata-rata.
* RMSE: LSTM (142.05) lebih kecil dibandingkan GRU (185.13), menunjukkan LSTM memiliki kesalahan yang lebih kecil pada kasus ekstrem.
* MAPE: LSTM (1.37%) lebih rendah dibanding GRU (1.52%), menegaskan bahwa LSTM memiliki tingkat kesalahan relatif yang lebih kecil terhadap nilai aktual.
* RSMPE: LSTM (1.41%) lebih kecil dari GRU (1.87%), memperkuat bahwa LSTM memberikan prediksi lebih stabil dibanding GRU.

**Kesimpulan:**

**LSTM lebih unggul dalam semua metrik dibandingkan GRU, menunjukkan bahwa model LSTM memberikan prediksi yang lebih akurat, stabil, dan responsif terhadap data. Oleh karena itu, LSTM adalah pilihan yang lebih baik untuk memprediksi harga saham BBCA.**

## Referensi

[[1] Otoritas Jasa Keuangan (OJK). "Pasar Modal." Diakses pada 16 Mei 2024.](https://www.ojk.go.id/id/kanal/pasar-modal/)

[[2] R. J. Hyndman dan G. Athanasopoulos, Forecasting: principles and practice. OTexts, 2018.](https://robjhyndman.com/uwafiles/fpp-notes.pdf)

[[3] Bursa Efek Indonesia (BEI). "Pengantar Analisis Teknikal." Diakses pada 16 Mei 2024.](https://www.idx.co.id/edukasi/artikel/pengantar-analisis-teknikal/)

[[4] Winarini. “Data 8 Saham dengan Kapitalisasi Pasar Terbesar (Big Caps) di BEI Per Januari 2024.” Dataindonesia.id, 2024.](https://dataindonesia.id/pasar-saham/detail/data-8-saham-dengan-kapitalisasi-pasar-terbesar-big-caps-di-bei-per-januari-2024)

[[5] A. Sharma, B. Singh, dan C. Kumar, "A Comparative Study of Stock Forecasts by LSTM and RNN Neural Networks".](https://www.researchgate.net/publication/340636297_Stock_Market_Prediction_Using_LSTM_Recurrent_Neural_Network)

[[6] D. Patel, E. Reddy, dan F. Shah, "Stock Market Analysis and Prediction Using LSTM: A Case Study on Technology Stocks"](https://www.researchgate.net/publication/379811995_Stock_Market_Analysis_and_Prediction_Using_LSTM_A_Case_Study_on_Technology_Stocks)