import os
import random
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import socket
import time
import csv
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import pandas as pd

# Koneksi soket untuk mengirim data ke ESP32
host = "192.168.4.1"
port = 80

# Fungsi untuk mengkonversi timestamp UNIX ke format waktu yang mudah dibaca
def format_readable_time(timestamp):
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime("%H:%M:%S.%f")[:-3]  # Format jam:menit:detik.milidetik

class SocketCommunicator:
    def __init__(self, host, port) -> None:
        self.host = host
        self.port = port
        self.socket = None
        self.connect()

    def connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((self.host, self.port))
            print("Terkoneksi dengan kursi roda")
            self.socket = s
        except socket.error:
            print("Tidak dapat terhubung ke kursi roda")

    def send(self, data):  # data sudah dalam bentuk bytes
        if self.socket:
            try:
                self.socket.send(data)
                return True
            except:
                print("Gagal mengirim data")
                return False
        return False

# Inisialisasi koneksi soket
s = SocketCommunicator(host, port)

print("Memuat model dan dataset...")
# Load model
model = tf.keras.models.load_model('gesture_recognition_model_var1.h5')

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Label encoder mapping
label_encoder = {
    0: 'Kanan',
    1: 'Kiri',
    2: 'Maju',
    3: 'Mundur',
    4: 'Stop',
}

# Fungsi normalisasi
def normalize_landmarks(landmarks):
    # Periksa apakah data bisa di-reshape ke (21, 3)
    if len(landmarks) != 63:  # 21 landmark * 3 koordinat (x,y,z)
        print(f"Melewati data dengan panjang {len(landmarks)} (seharusnya 63)")
        return None
    
    try:
        landmarks = np.array(landmarks).reshape((21, 3))
        origin = landmarks[0]
        landmarks -= origin
        scale = np.linalg.norm(landmarks[0] - landmarks[5])
        if scale > 0:
            landmarks /= scale
        return landmarks.flatten().tolist()
    except Exception as e:
        print(f"Error dalam normalisasi landmark: {e}")
        return None

# Fungsi untuk memuat dataset dan normalisasi data
def load_dataset(dataset_path, gestures, num_sequences, sequence_length):
    X, y = [], []
    skipped_sequences = 0
    
    for label, gesture in enumerate(gestures):
        for seq in range(num_sequences):
            sequence_folder = os.path.join(dataset_path, gesture, f'sequence_{seq}')
            
            # Periksa apakah folder sequence ada
            if not os.path.exists(sequence_folder):
                print(f"Folder tidak ditemukan: {sequence_folder}")
                skipped_sequences += 1
                continue
                
            data_path = os.path.join(sequence_folder, 'data.npy')
            if os.path.exists(data_path):
                try:
                    sequence_data = np.load(data_path)  # Memuat file .npy
                    
                    # Cetak bentuk data untuk debugging
                    print(f"Sequence {gesture}-{seq} shape: {sequence_data.shape}")
                    
                    # Periksa jika data kita sudah dalam bentuk sequence
                    if len(sequence_data.shape) == 2 and sequence_data.shape[0] == sequence_length:
                        # Ini adalah data sequence (10, 63)
                        processed_sequence = []
                        
                        # Proses tiap frame dalam sequence
                        for frame_idx in range(sequence_length):
                            frame_data = sequence_data[frame_idx]  # Ambil frame ke-i
                            
                            # Periksa apakah frame memiliki 63 nilai (21 landmark × 3 koordinat)
                            if len(frame_data) != 63:
                                print(f"Frame {frame_idx} dari {gesture}-{seq} memiliki ukuran tidak valid: {len(frame_data)}")
                                continue
                                
                            # Normalisasi frame
                            normalized = normalize_landmarks(frame_data)
                            if normalized is not None:
                                processed_sequence.append(normalized)
                        
                        # Jika sequence sudah diproses dengan benar
                        if len(processed_sequence) == sequence_length:
                            X.append(np.array(processed_sequence))
                            y.append(label)
                        else:
                            print(f"Sequence {gesture}-{seq} tidak memiliki jumlah frame yang valid: {len(processed_sequence)}/{sequence_length}")
                            skipped_sequences += 1
                    else:
                        print(f"Data bentuk tidak sesuai untuk {gesture}-{seq}: {sequence_data.shape}, diharapkan ({sequence_length}, 63)")
                        skipped_sequences += 1
                        
                except Exception as e:
                    print(f"Error memproses {data_path}: {e}")
                    skipped_sequences += 1
                    continue
            else:
                print(f"File data tidak ditemukan: {data_path}")
                skipped_sequences += 1
                continue
    
    print(f"Total {skipped_sequences} sequences dilewati karena bentuk data tidak sesuai")
    
    # Jika tidak ada data yang valid
    if len(X) == 0:
        raise ValueError("Tidak ada data valid yang dapat dimuat!")
        
    # Data sudah dalam bentuk (samples, timesteps, features)
    X = np.array(X)
    y = np.array(y)
    
    print(f"Data dimuat: X shape: {X.shape}, y shape: {y.shape}")
    return X, y

# Konfigurasi dataset dan model
DATASET_PATH = 'dataset_testing'      # Path dataset testing baru
gestures = ['Kanan', 'Kiri', 'Maju', 'Mundur', 'Stop']
num_classes = len(gestures)
sequence_length = 10  # Panjang sequence (10 frame)
num_sequences = 11    # Jumlah sequence per kelas

try:
    # Memuat dataset
    X_test, y_test = load_dataset(DATASET_PATH, gestures, num_sequences, sequence_length)
    
    # Convert ke one-hot encoding
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
    
    # Pastikan model dikompilasi dengan metrik yang tepat
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Evaluasi model
    loss, acc = model.evaluate(X_test, y_test_cat)
    print(f"Akurasi: {acc:.4f} | Loss: {loss:.4f}")
    
    # Prediksi menggunakan model
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Membuat folder hasil evaluasi
    folder_name = datetime.datetime.now().strftime("EvaluasiTestingBaru_%Y%m%d_%H%M%S")
    os.makedirs(folder_name, exist_ok=True)
    
    # Membuat Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=gestures)
    fig_cm, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.title("Confusion Matrix - Dataset Testing")
    plt.savefig(os.path.join(folder_name, "confusion_matrix_testing_baru.png"))
    plt.close(fig_cm)
    
    # Membuat Classification Report
    report = classification_report(y_test, y_pred, target_names=gestures, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(folder_name, "classification_report_testing_baru.csv"))
    
    # Fungsi untuk menghitung TP, TN, FP, FN
    def calculate_confusion_elements(y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred)
        metrics = []
        for i, class_name in enumerate(classes):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - (TP + FP + FN)
            metrics.append({'Class': class_name, 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN})
        return pd.DataFrame(metrics)
    
    # Menyimpan hasil TP, TN, FP, FN ke CSV
    conf_matrix_df = calculate_confusion_elements(y_test, y_pred, gestures)
    conf_matrix_df.to_csv(os.path.join(folder_name, "TP_TN_FP_FN_testing_baru.csv"), index=False)
    
    # Simpan juga raw predictions
    prediction_df = pd.DataFrame({
        'True_Label': [gestures[y] for y in y_test],
        'Predicted_Label': [gestures[y] for y in y_pred]
    })
    prediction_df.to_csv(os.path.join(folder_name, "predictions.csv"), index=False)
    
    print(f"Hasil evaluasi disimpan dalam folder: {folder_name}")

except Exception as e:
    print(f"Error dalam eksekusi: {e}")
    import traceback
    traceback.print_exc()