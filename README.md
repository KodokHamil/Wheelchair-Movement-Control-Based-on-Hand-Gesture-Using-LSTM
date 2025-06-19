# Wheelchair Movement Control Based on Hand Gesture Using LSTM

Sistem ini mengendalikan **kursi roda** menggunakan **gestur tangan**, yang dikenali melalui model **LSTM** berbasis landmark dari **MediaPipe**, lalu hasil prediksi dikirim ke **ESP32** via **socket TCP** untuk mengontrol arah gerak kursi roda.

---

## 🧠 Fitur Utama

- **Pengumpulan Data Gesture**
  - Rekam gestur tangan (`Maju`, `Mundur`, `Kanan`, `Kiri`, `Stop`) secara real-time.
  - Landmark tangan dikumpulkan dalam bentuk urutan 10 frame.

- **Pelatihan Model**
  - Gunakan `data_training.py` untuk melatih model LSTM dan menyimpannya sebagai `.h5`.

- **Pengujian dan Kontrol Kursi Roda**
  - Jalankan `kursiroda.py` untuk:
    - Prediksi gesture secara real-time.
    - Kirim sinyal kontrol ke **ESP32**.
    - Simpan log ke `.csv` dan summary ke `.txt`.
    - Tiga mode pengujian: **Manual**, **Auto Sequential**, dan **Free Testing**.

---

## 🔧 Instalasi

1. Clone repository ini:
   ```bash
   git clone https://github.com/KodokHamil/Wheelchair-Movement-Control-Based-on-Hand-Gesture-Using-LSTM.git
   cd Wheelchair-Movement-Control-Based-on-Hand-Gesture-Using-LSTM/finished
   ```

2. Buat virtual environment dan install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

   Isi `requirements.txt`:
   ```
   opencv-python
   mediapipe
   tensorflow
   numpy
   scikit-learn
   ```

---

## 🚦 Cara Penggunaan

### 1. Kumpulkan Data Gesture
```bash
python dataset_collect.py
```
- Tekan `r` untuk mulai merekam, `q` untuk keluar.
- Data disimpan otomatis per kelas gesture.

### 2. Latih Model
```bash
python data_training.py
```
- Model akan disimpan sebagai `gesture_recognition_model_*.h5`.

### 3. Uji & Kontrol Kursi Roda
```bash
python kursiroda.py
```
- Akan muncul pilihan mode:
  1. **Manual Mode** – Masukkan label gesture secara manual untuk evaluasi akurasi.
  2. **Auto Sequential Mode** – Testing otomatis semua gesture berurutan, 50 sampel/gesture.
  3. **Free Testing** – Hanya prediksi, tidak mencatat akurasi.

ESP32 akan menerima karakter kontrol:
| Gesture  | Karakter yang dikirim |
|----------|------------------------|
| Kanan    | `E\n`                 |
| Kiri     | `A\n`                 |
| Maju     | `B\n`                 |
| Mundur   | `D\n`                 |
| Stop     | `C\n`                 |

---

## 📦 Struktur File

```
├── dataset_collect.py                # Rekam data gesture
├── data_training.py                 # Pelatihan model LSTM
├── kursiroda.py                     # Prediksi real-time + kontrol ESP32
├── testmodel.py                     # Cek prediksi gesture tanpa ESP32
├── gesture_recognition_model_var3.h5  # Model yang digunakan
├── gesture_log_*.csv                # Log pengujian realtime
├── accuracy_summary_*.txt           # Ringkasan akurasi
└── requirements.txt
```

---

## 📄 Contoh Log CSV

| start_gesture_time | end_gesture_time | predicted_label | true_label | is_correct | send_time | inference_time | fps | confidence |
|--------------------|------------------|------------------|------------|------------|-----------|----------------|-----|------------|
| 19:16:59.735       | 19:16:59.786     | Maju             | Maju       | True       | 19:16:59.800 | 0.0512        | 27  | 0.9934     |

---

## ⚙️ Penjelasan `kursiroda.py`

- **Prediksi dilakukan setiap 10 frame gesture**.
- **Tiga mode pengujian**:
  - **Manual Mode**: tekan `t` untuk input label ground truth.
  - **Auto Mode**: sistem memberi instruksi gesture, countdown, lalu testing otomatis.
  - **Free Mode**: hanya tampilkan hasil prediksi dan kirim ke ESP32 tanpa penilaian.

- Semua prediksi dicatat (jika akurasi diukur) ke file `.csv`, dan ringkasan akurasi akan disimpan ke `.txt`.

- **Kode socket**:
  ```python
  host = "192.168.4.1"
  port = 80
  s = SocketCommunicator(host, port)
  s.send(b'A\n')  # contoh: kirim gesture 'Kiri'
  ```

- **Logging mencatat**:
  - Waktu mulai/akhir gesture
  - Label prediksi & ground truth
  - Waktu inferensi
  - FPS
  - Confidence score (softmax)

---

## 🧠 Evaluasi Performa

- Akurasi model dan statistik per gesture ditampilkan langsung di console.
- Akurasi total ditampilkan setelah pengujian selesai.
- Semua data disimpan untuk analisis lebih lanjut.

---

## 🛠️ Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- TensorFlow
- NumPy
- scikit-learn

---

## 📡 Komunikasi dengan ESP32

ESP32 harus dalam mode Access Point (`192.168.4.1`) dan mendengarkan port `80`. Server ESP32 harus menerima 1 karakter ASCII untuk kontrol arah, misalnya:

```c++
if (incoming == 'A') { // KIRI }
if (incoming == 'B') { // MAJU }
if (incoming == 'C') { // STOP }
```

---

## 👨‍💻 Penulis

- **Nama**: Andrya Muhammad Naufal  
- 📧 Email: andryanaufal@gmail.com  
- 🔗 GitHub: [@KodokHamil](https://github.com/KodokHamil)

---

## 📝 Lisensi

MIT License © 2025 Andrya Muhammad Naufal
