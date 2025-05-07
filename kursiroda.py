import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import socket
import time

# Koneksi soket untuk mengirim data ke ESP32
host = "192.168.4.1"
port = 80

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
            self.socket.send(data)

# Inisialisasi koneksi soket
s = SocketCommunicator(host, port)

# Load model
model = tf.keras.models.load_model('gesture_recognition_model_4.h5')

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
    landmarks = np.array(landmarks).reshape((21, 3))
    origin = landmarks[0]
    landmarks -= origin
    scale = np.linalg.norm(landmarks[0] - landmarks[5])
    if scale > 0:
        landmarks /= scale
    return landmarks.flatten().tolist()

# Kamera
cap = cv2.VideoCapture(0)

prev_frame_time = 0
frame_sequence = []

print("Mulai pengujian... Tekan 'q' untuk keluar")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    new_frame_time = time.time()

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

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
                input_sequence = np.array([frame_sequence])
                prediction = model.predict(input_sequence, verbose=0)
                predicted_label_index = np.argmax(prediction)
                predicted_label = label_encoder[predicted_label_index]

                cv2.putText(image, f"Gesture: {predicted_label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Kirim ke ESP32 sesuai gesture
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

    # FPS
    fps = 1 / (new_frame_time - prev_frame_time) if new_frame_time - prev_frame_time != 0 else 0
    prev_frame_time = new_frame_time
    cv2.putText(image, f'FPS: {int(fps)}', (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Testing Gesture Recognition", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
