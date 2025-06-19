import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import socket
import time
import csv
import datetime

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

# Load model
model = tf.keras.models.load_model('gesture_recognition_model_var3.h5')

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

# OPSI 1: Testing Mode dengan Sequence Otomatis
class AutoTester:
    def __init__(self):
        self.gesture_sequence = ['Maju', 'Mundur', 'Kanan', 'Kiri', 'Stop']
        self.current_gesture_index = 0
        self.samples_per_gesture = 50  # Diubah menjadi 50 sample
        self.current_sample_count = 0
        self.countdown_timer = 0
        self.preparation_time = 3  # 3 detik persiapan
        self.is_testing = False
        self.is_preparing = False
        self.last_prediction_time = 0
        self.prediction_interval = 0.2  # Ambil prediksi setiap 0.2 detik
        
    def get_current_true_label(self):
        if self.is_testing:
            return self.gesture_sequence[self.current_gesture_index]
        return None
    
    def start_next_gesture(self):
        if self.current_gesture_index < len(self.gesture_sequence):
            self.is_preparing = True
            self.countdown_timer = time.time() + self.preparation_time
            self.current_sample_count = 0
            print(f"\n=== PERSIAPAN GESTURE: {self.gesture_sequence[self.current_gesture_index]} ===")
            print(f"Bersiap dalam {self.preparation_time} detik...")
            
    def update(self):
        current_time = time.time()
        
        if self.is_preparing:
            remaining = self.countdown_timer - current_time
            if remaining <= 0:
                self.is_preparing = False
                self.is_testing = True
                print(f"MULAI! Lakukan gesture: {self.gesture_sequence[self.current_gesture_index]}")
                print(f"Target: {self.samples_per_gesture} samples")
            else:
                print(f"Persiapan: {remaining:.1f}s - {self.gesture_sequence[self.current_gesture_index]}", end='\r')
        
        elif self.is_testing:
            if self.current_sample_count >= self.samples_per_gesture:
                self.is_testing = False
                self.current_gesture_index += 1
                print(f"\nGesture {self.gesture_sequence[self.current_gesture_index-1]} selesai!")
                
                if self.current_gesture_index < len(self.gesture_sequence):
                    time.sleep(1)  # Jeda sebentar
                    self.start_next_gesture()
                else:
                    print("\n=== SEMUA GESTURE TESTING SELESAI ===")
                    return False  # Testing selesai
        return True
    
    def should_record_prediction(self):
        """Cek apakah sudah waktunya untuk record prediksi (setiap 0.2 detik)"""
        current_time = time.time()
        if self.is_testing and (current_time - self.last_prediction_time) >= self.prediction_interval:
            self.last_prediction_time = current_time
            return True
        return False
    
    def record_prediction(self):
        if self.is_testing:
            self.current_sample_count += 1
            remaining = self.samples_per_gesture - self.current_sample_count
            progress = (self.current_sample_count / self.samples_per_gesture) * 100
            if remaining > 0:
                print(f"Sample {self.current_sample_count}/{self.samples_per_gesture} ({progress:.1f}%) - Sisa: {remaining}")
            return True
        return False

# OPSI 2: Load dari file dataset untuk validasi
def load_validation_data(csv_file_path):
    """
    Memuat data validasi dari file CSV
    Format CSV: gesture_label, frame1_data, frame2_data, ..., frame10_data
    """
    validation_data = []
    try:
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                true_label = row[0]
                # Assuming the rest are landmark data
                validation_data.append(true_label)
        return validation_data
    except FileNotFoundError:
        print(f"File validasi {csv_file_path} tidak ditemukan")
        return None

# Pilih mode testing
print("Pilih mode testing:")
print("1. Manual (tekan 't' untuk set true label)")
print("2. Auto Sequential (otomatis berurutan: Maju->Mundur->Kanan->Kiri->Stop)")
print("3. Free Testing (hanya prediksi, tanpa akurasi)")

mode = input("Pilih mode (1-3): ").strip()

# Inisialisasi berdasarkan mode
auto_tester = None
if mode == "2":
    auto_tester = AutoTester()
    auto_tester.start_next_gesture()

# Variabel untuk tracking akurasi
total_predictions = 0
correct_predictions = 0
current_true_label = None

# Statistik per gesture
gesture_stats = {
    'Maju': {'total': 0, 'correct': 0},
    'Mundur': {'total': 0, 'correct': 0},
    'Kanan': {'total': 0, 'correct': 0},
    'Kiri': {'total': 0, 'correct': 0},
    'Stop': {'total': 0, 'correct': 0}
}

# Fungsi untuk mendapatkan input true label dari user (untuk mode manual)
def get_true_label_input():
    print("\nMasukkan true label untuk gesture yang akan Anda lakukan:")
    print("1. Kanan")
    print("2. Kiri") 
    print("3. Maju")
    print("4. Mundur")
    print("5. Stop")
    print("0. Skip prediction (tidak dihitung akurasi)")
    
    while True:
        try:
            choice = input("Pilih (0-5): ").strip()
            if choice == '0':
                return None
            elif choice == '1':
                return 'Kanan'
            elif choice == '2':
                return 'Kiri'
            elif choice == '3':
                return 'Maju'
            elif choice == '4':
                return 'Mundur'
            elif choice == '5':
                return 'Stop'
            else:
                print("Pilihan tidak valid. Masukkan 0-5.")
        except:
            print("Input tidak valid. Masukkan 0-5.")

# Fungsi normalisasi
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape((21, 3))
    origin = landmarks[0]
    landmarks -= origin
    scale = np.linalg.norm(landmarks[0] - landmarks[5])
    if scale > 0:
        landmarks /= scale
    return landmarks.flatten().tolist()

# Fungsi untuk menampilkan statistik akurasi
def display_accuracy_stats():
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\n=== STATISTIK AKURASI KESELURUHAN ===")
        print(f"Total Prediksi: {total_predictions}")
        print(f"Prediksi Benar: {correct_predictions}")
        print(f"Prediksi Salah: {total_predictions - correct_predictions}")
        print(f"Akurasi: {accuracy:.2f}%")
        
        print(f"\n=== STATISTIK PER GESTURE ===")
        for gesture, stats in gesture_stats.items():
            if stats['total'] > 0:
                gesture_acc = (stats['correct'] / stats['total']) * 100
                print(f"{gesture}: {stats['correct']}/{stats['total']} = {gesture_acc:.2f}%")
        print("=" * 40)
        return accuracy
    return 0

# Kamera
cap = cv2.VideoCapture(0)

prev_frame_time = 0
frame_sequence = []

# CSV to log events and times
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_log = open(f'gesture_log_{timestamp}.csv', 'w', newline='')
csv_writer = csv.writer(csv_log)
csv_writer.writerow([
    'start_gesture_time', 
    'end_gesture_time', 
    'predicted_label',
    'true_label',
    'is_correct',
    'send_time', 
    'inference_time',
    'fps',
    'confidence'
])

print(f"\nMode yang dipilih: {mode}")
if mode == "1":
    print("Mode Manual - Tekan 't' untuk set true label")
elif mode == "2":
    print("Mode Auto Sequential - Testing otomatis berurutan")
elif mode == "3":
    print("Mode Free Testing - Hanya prediksi")

print("Tekan 'q' untuk keluar, 's' untuk statistik")

# Main loop
running = True
while cap.isOpened() and running:
    ret, frame = cap.read()
    if not ret:
        break

    # Update auto tester jika menggunakan mode 2
    if mode == "2" and auto_tester:
        if not auto_tester.update():
            running = False  # Testing selesai
            break
        current_true_label = auto_tester.get_current_true_label()

    new_frame_time = time.time()
    
    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time) if new_frame_time - prev_frame_time != 0 else 0
    prev_frame_time = new_frame_time

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    start_gesture_time = None
    end_gesture_time = None
    send_time = None
    gesture_detected = None
    inference_time = None
    is_correct = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            raw_landmarks = []
            for lm in hand_landmarks.landmark:
                raw_landmarks.extend([lm.x, lm.y, lm.z])
            normalized = normalize_landmarks(raw_landmarks)

            frame_sequence.append(normalized)
            frame_sequence = frame_sequence[-10:]  # pastikan selalu 10 frame

            if len(frame_sequence) == 10:
                # Start Gesture Detection
                start_gesture_time = time.time()

                input_sequence = np.array([frame_sequence])
                prediction = model.predict(input_sequence, verbose=0)
                predicted_label_index = np.argmax(prediction)
                predicted_label = label_encoder[predicted_label_index]
                confidence_score = float(np.max(prediction))
                end_gesture_time = time.time()
                
                # Calculate inference time
                inference_time = end_gesture_time - start_gesture_time
                
                gesture_detected = predicted_label

                # Check accuracy jika ada true label
                should_record = True
                if mode == "2" and auto_tester:
                    should_record = auto_tester.should_record_prediction()
                
                if current_true_label is not None and mode != "3" and should_record:
                    total_predictions += 1
                    is_correct = (predicted_label == current_true_label)
                    
                    # Update statistik per gesture
                    gesture_stats[current_true_label]['total'] += 1
                    if is_correct:
                        correct_predictions += 1
                        gesture_stats[current_true_label]['correct'] += 1
                    
                    # Record prediction untuk auto tester
                    if mode == "2" and auto_tester:
                        auto_tester.record_prediction()
                    
                    # Reset untuk mode manual
                    if mode == "1":
                        current_true_label = None

                cv2.putText(image, f"Predicted: {predicted_label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # Tampilkan true label jika ada
                if current_true_label:
                    cv2.putText(image, f"True: {current_true_label}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Display inference time
                cv2.putText(image, f"Inference: {inference_time:.5f} s", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # Display accuracy
                if total_predictions > 0:
                    current_accuracy = (correct_predictions / total_predictions) * 100
                    cv2.putText(image, f"Accuracy: {current_accuracy:.1f}%", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Kirim ke ESP32 sesuai gesture
                send_time = time.time()
                if predicted_label == 'Kanan':
                    s.send(b'E\n')
                elif predicted_label == 'Kiri':
                    s.send(b'A\n')
                elif predicted_label == 'Maju':
                    s.send(b'B\n')
                elif predicted_label == 'Mundur':
                    s.send(b'D\n')
                elif predicted_label == 'Stop':
                    s.send(b'C\n')

                # Log to CSV - hanya log yang direcord untuk akurasi
                if all([start_gesture_time, end_gesture_time, send_time, inference_time, gesture_detected]):
                    # Untuk mode auto, hanya log yang digunakan untuk akurasi
                    if mode == "2" and auto_tester and not should_record:
                        pass  # Skip logging jika tidak direcord untuk akurasi
                    else:
                        csv_writer.writerow([
                            format_readable_time(start_gesture_time),
                            format_readable_time(end_gesture_time),
                            gesture_detected,
                            current_true_label if current_true_label else "N/A",
                            is_correct if is_correct is not None else "N/A",
                            format_readable_time(send_time),
                            f"{inference_time:.5f}",
                            f"{fps:.2f}",
                            f"{confidence_score:.4f}"
                        ])
                        
                        csv_log.flush()
                        
                        # Console output
                        if is_correct is not None:
                            status = "✓ BENAR" if is_correct else "✗ SALAH"
                            print(f"Predicted: {gesture_detected}, True: {current_true_label}, {status}, Inference: {inference_time:.5f}s")
                        else:
                            print(f"Predicted: {gesture_detected}, Inference: {inference_time:.5f}s")

    # Display FPS and mode info
    cv2.putText(image, f'FPS: {int(fps)}', (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    if mode == "1":
        cv2.putText(image, "Press 't' for true label", (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    elif mode == "2" and auto_tester:
        if auto_tester.is_preparing:
            remaining_prep = auto_tester.countdown_timer - time.time()
            cv2.putText(image, f"PREPARING: {auto_tester.gesture_sequence[auto_tester.current_gesture_index]} ({remaining_prep:.1f}s)", 
                       (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif auto_tester.is_testing:
            progress = (auto_tester.current_sample_count / auto_tester.samples_per_gesture) * 100
            cv2.putText(image, f"TESTING: {auto_tester.current_sample_count}/{auto_tester.samples_per_gesture} ({progress:.1f}%)", 
                       (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Gesture: {auto_tester.gesture_sequence[auto_tester.current_gesture_index]}", 
                       (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Testing Gesture Recognition", image)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t') and mode == "1":
        current_true_label = get_true_label_input()
        if current_true_label:
            print(f"True label diset ke: {current_true_label}")
        else:
            print("Tidak menggunakan true label untuk gesture selanjutnya")
    elif key == ord('s'):
        display_accuracy_stats()

cap.release()
cv2.destroyAllWindows()
csv_log.close()

# Tampilkan statistik final
print(f"\nLog tersimpan di: gesture_log_{timestamp}.csv")
final_accuracy = display_accuracy_stats()

# Simpan summary akurasi ke file terpisah
summary_file = f'accuracy_summary_{timestamp}.txt'
with open(summary_file, 'w') as f:
    f.write(f"SUMMARY AKURASI MODEL GESTURE RECOGNITION\n")
    f.write(f"Mode Testing: {mode}\n")
    f.write(f"Tanggal: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"STATISTIK KESELURUHAN:\n")
    f.write(f"Total Prediksi: {total_predictions}\n")
    f.write(f"Prediksi Benar: {correct_predictions}\n")
    f.write(f"Prediksi Salah: {total_predictions - correct_predictions}\n")
    f.write(f"Akurasi: {final_accuracy:.2f}%\n\n")
    
    f.write(f"STATISTIK PER GESTURE:\n")
    for gesture, stats in gesture_stats.items():
        if stats['total'] > 0:
            gesture_acc = (stats['correct'] / stats['total']) * 100
            f.write(f"{gesture}: {stats['correct']}/{stats['total']} = {gesture_acc:.2f}%\n")

print(f"Summary akurasi tersimpan di: {summary_file}")