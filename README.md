# Wheelchair Movement Control Based on Hand Gesture Using LSTM

## Deskripsi
Proyek ini bertujuan untuk mengendalikan **kursi roda** menggunakan **gestur tangan** yang diidentifikasi dengan menggunakan model **LSTM (Long Short-Term Memory)**. Sistem ini memanfaatkan **MediaPipe** untuk mendeteksi dan mengekstrak fitur landmark tangan, kemudian menggunakan model LSTM untuk memprediksi gestur. Hasil prediksi dikirim ke **ESP32** melalui komunikasi **socket**, yang selanjutnya mengendalikan pergerakan kursi roda.

---

## Library yang Digunakan
- **OpenCV** (`cv2`): Untuk pemrosesan gambar dan video, serta pengambilan input dari kamera.
- **NumPy** (`np`): Untuk manipulasi data numerik dan array.
- **MediaPipe** (`mediapipe`): Untuk deteksi landmark tangan yang digunakan dalam pengenalan gestur.
- **TensorFlow** (`tensorflow`): Untuk membangun dan melatih model **LSTM** dalam pengenalan gestur.
- **scikit-learn** (`sklearn`): Untuk preprocessing data, evaluasi model (misalnya, confusion matrix), dan pembagian dataset.
- **Socket**: Untuk mengirim data hasil prediksi ke **ESP32**, yang mengontrol kursi roda.

---

## Fitur
- **Pengumpulan Data Gesture**:
  - Pengguna dapat merekam data gestur tangan menggunakan kamera dan MediaPipe.
  - Setiap gesture yang direkam akan disimpan sebagai urutan (sequence) frame dan landmark tangan yang distandarisasi.

- **Model LSTM**:
  - Model LSTM dilatih menggunakan dataset gesture untuk memprediksi **5 jenis gestur tangan**: `Maju`, `Mundur`, `Kanan`, `Kiri`, `Stop`.
  - Model kemudian digunakan untuk melakukan prediksi secara real-time.

- **Kontrol Kursi Roda**:
  - Setelah model memprediksi gestur, hasil prediksi akan dikirimkan ke **ESP32** menggunakan socket untuk mengontrol arah kursi roda (misalnya, `Kanan`, `Kiri`, `Maju`, `Mundur`, `Stop`).

---

## Cara Penggunaan

### 1. **Mempersiapkan Dataset**
Untuk mengumpulkan dataset gesture, jalankan script `dataset_collect.py`. Program ini akan menangkap video dari kamera dan mengumpulkan **landmark tangan** untuk setiap gesture yang kamu tentukan. Tekan `r` untuk mulai merekam dan `q` untuk keluar.

### 2. **Melatih Model LSTM**
Setelah data terkumpul, gunakan script `data_training.py` untuk melatih model LSTM berdasarkan data yang telah dikumpulkan. Model ini akan menyimpan hasil pelatihan dalam file `.h5`.

### 3. **Testing Model**
Untuk menguji model yang telah dilatih, jalankan script `testmodel.py`. Program ini akan menggunakan model untuk melakukan prediksi gesture secara real-time menggunakan kamera.

### 4. **Integrasi dengan Kursi Roda**
Jika sistem sudah berjalan dengan baik, jalankan `kursiroda.py` untuk menghubungkan hasil prediksi dengan **ESP32** menggunakan socket, yang akan mengontrol pergerakan kursi roda berdasarkan gestur yang terdeteksi.

---

## Catatan
- Pastikan model yang digunakan dalam script **testmodel.py** sesuai dengan model yang sudah dilatih (`gesture_recognition_model.h5`).
- Proyek ini memerlukan koneksi jaringan untuk komunikasi dengan **ESP32**.

---

## Penulis
- **Nama:** [Nama Kamu]
- **Email:** [Email Kamu]
- **GitHub:** [Link GitHub Kamu]
