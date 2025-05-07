import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load model yang telah dilatih
model = tf.keras.models.load_model('gesture_recognition_model_2.h5')

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

# Setup Kamera
cap = cv2.VideoCapture(0)

# Menampilkan pesan
print("Mulai pengujian... Tekan 'q' untuk keluar")

# Untuk mengumpulkan data urutan (sequence) dari frame kamera
frame_sequence = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

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

    # Menampilkan frame dengan hasil prediksi
    cv2.imshow("Testing Gesture Recognition", image)

    # Menunggu input pengguna
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
