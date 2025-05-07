import cv2
import numpy as np
import os
import mediapipe as mp

# === Fungsi Normalisasi Landmark ===
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape((21, 3))
    origin = landmarks[0]
    landmarks -= origin
    scale = np.linalg.norm(landmarks[0] - landmarks[5])
    if scale > 0:
        landmarks /= scale
    return landmarks.flatten().tolist()

# === Setup MediaPipe ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === Pengaturan Dataset ===
DATA_PATH = 'dataset_tangan_kanan'
gestures = ['Maju', 'Mundur', 'Kanan', 'Kiri', 'Stop', 'NA']
current_gesture = 'NA'
sequence_length = 10
sequence_counter = 0

# === Setup Folder Dataset ===
os.makedirs(os.path.join(DATA_PATH, current_gesture), exist_ok=True)

# === Setup Kamera ===
cap = cv2.VideoCapture(0)
collecting = False
frame_counter = 0
data = []
frames_to_save = []  # Untuk simpan gambar asli

print(f"Tekan 'r' untuk mulai rekam 1 sequence ({sequence_length} frame) gesture: {current_gesture}")
print(f"Tekan 'q' untuk keluar")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if handedness.classification[0].label == 'Right':
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                raw_landmarks = []
                for lm in hand_landmarks.landmark:
                    raw_landmarks.extend([lm.x, lm.y, lm.z])
                normalized = normalize_landmarks(raw_landmarks)

                if collecting:
                    data.append(normalized)
                    frames_to_save.append(image.copy())  # Simpan frame saat ini
                    frame_counter += 1
                    cv2.putText(image, f"Recording: {frame_counter}/{sequence_length}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if frame_counter == sequence_length:
                        save_path = os.path.join(DATA_PATH, current_gesture, f'sequence_{sequence_counter}')
                        os.makedirs(save_path, exist_ok=True)

                        # Simpan landmark
                        np.save(os.path.join(save_path, 'data.npy'), np.array(data))

                        # Simpan semua frame sebagai .jpg
                        for i, frame_img in enumerate(frames_to_save):
                            frame_filename = os.path.join(save_path, f'frame_{i:02d}.jpg')
                            cv2.imwrite(frame_filename, frame_img)

                        print(f"Sequence {sequence_counter} saved: {save_path}")
                        sequence_counter += 1
                        data = []
                        frames_to_save = []
                        frame_counter = 0
                        collecting = False

    cv2.putText(image, f'Gesture: {current_gesture}  Seq: {sequence_counter}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("MediaPipe - Rekam Tangan Kanan", image)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('r'):
        if not collecting:
            print(f"Mulai rekam sequence {sequence_counter}...")
            collecting = True
            frame_counter = 0
            data = []
            frames_to_save = []
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
