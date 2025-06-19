import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import datetime

# Load model yang telah dilatih
model = tf.keras.models.load_model('gesture_recognition_model_var1.h5')

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Label encoder yang digunakan untuk mengubah label
label_encoder = {
    0: 'Kanan',
    1: 'Kiri',
    2: 'Maju',
    3: 'Mundur',
    4: 'Stop',
}

# Fungsi untuk normalisasi landmark
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape((21, 3))
    origin = landmarks[0]
    landmarks -= origin
    scale = np.linalg.norm(landmarks[0] - landmarks[5])
    if scale > 0:
        landmarks /= scale
    return landmarks.flatten().tolist()

# Estimasi jarak berdasarkan lebar tangan
KNOWN_HAND_WIDTH_CM = 10.0  # lebar tangan rata-rata (cm)
FOCAL_LENGTH_PX = 600.0     # sesuaikan dengan kamera Anda

# Setup Kamera
cap = cv2.VideoCapture(0)

# Menampilkan pesan
print("Mulai pengujian... Tekan 'q' untuk keluar")

# Untuk mengumpulkan data urutan (sequence) dari frame kamera
frame_sequence = []

prev_frame_time = 0  # Untuk menghitung FPS

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    new_frame_time = time.time()

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Menghitung FPS
    fps = 1 / (new_frame_time - prev_frame_time) if new_frame_time - prev_frame_time != 0 else 0
    prev_frame_time = new_frame_time

    # Menambahkan FPS ke layar
    cv2.putText(image, f'FPS: {int(fps)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Menampilkan landmark tangan
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Normalisasi landmark
            raw_landmarks = []
            for lm in hand_landmarks.landmark:
                raw_landmarks.extend([lm.x, lm.y, lm.z])
            normalized = normalize_landmarks(raw_landmarks)

            # Menambahkan frame saat ini ke dalam sequence
            frame_sequence.append(normalized)

            # Jika urutan mencapai panjang yang diinginkan, lakukan prediksi
            if len(frame_sequence) == 10:
                # Mengubah frame sequence menjadi format yang sesuai dengan input LSTM
                input_sequence = np.array([frame_sequence])  # Membuat dimensi (1, 10, 63)

                # Melakukan prediksi gesture
                prediction = model.predict(input_sequence)
                predicted_label_index = np.argmax(prediction)
                predicted_label = label_encoder[predicted_label_index]

                # Menampilkan hasil prediksi
                cv2.putText(image, f"Gesture: {predicted_label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Menghapus frame pertama untuk memastikan sequence selalu 10 frame
                frame_sequence.pop(0)

                # --- PENGUKURAN JARAK ---
                # Mengukur jarak tangan dari kamera berdasarkan lebar tangan
                hand_landmarks_px = []
                for lm in hand_landmarks.landmark:
                    hand_landmarks_px.append([lm.x * image.shape[1], lm.y * image.shape[0]])
                hand_landmarks_px = np.array(hand_landmarks_px)
                
                # Mengukur lebar tangan berdasarkan jarak dari ibu jari ke kelingking
                thumb_tip = hand_landmarks_px[4]  # Ujung ibu jari
                pinky_tip = hand_landmarks_px[20]  # Ujung kelingking
                hand_width_px = np.linalg.norm(thumb_tip - pinky_tip)
                
                if hand_width_px > 0:
                    distance_cm = (KNOWN_HAND_WIDTH_CM * FOCAL_LENGTH_PX) / hand_width_px
                    # Menampilkan jarak
                    cv2.putText(image, f'Dist: {distance_cm:.1f} cm', (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                # Menampilkan confidence score
                confidence = np.max(prediction)
                cv2.putText(image, f'Conf: {confidence:.2f}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Menampilkan frame dengan hasil prediksi
    cv2.imshow("Testing Gesture Recognition", image)

    # Menunggu input pengguna
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()