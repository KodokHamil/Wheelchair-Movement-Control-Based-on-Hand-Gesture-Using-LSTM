import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# === Fungsi untuk Menentukan Nama File Unik ===
def get_unique_model_filename(base_filename):
    counter = 1
    filename = base_filename
    while os.path.exists(filename):
        filename = base_filename.replace('.h5', f'_{counter}.h5')
        counter += 1
    return filename

# === Menyiapkan Dataset ===
DATA_PATH = 'dataset_tangan_kanan'
gestures = ['Maju', 'Mundur', 'Kanan', 'Kiri', 'Stop']
sequences = []
labels = []

for gesture in gestures:
    gesture_folder = os.path.join(DATA_PATH, gesture)
    for seq_folder in os.listdir(gesture_folder):
        seq_data = np.load(os.path.join(gesture_folder, seq_folder, 'data.npy'))
        sequences.append(seq_data)
        labels.append(gesture)

# === Encode Labels ===
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# === Konversi ke Format LSTM ===
X = np.array(sequences)
y = np.array(labels)

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# === Model LSTM ===
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(gestures), activation='softmax'))

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# === Training ===
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# === Evaluasi ===
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# === Simpan Model ===
model_filename = get_unique_model_filename('gesture_recognition_model.h5')
model.save(model_filename)
print(f"Model disimpan dengan nama: {model_filename}")

# === Visualisasi Akurasi dan Loss ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Akurasi Training dan Validasi')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Training dan Validasi')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# === Confusion Matrix ===
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=gestures, yticklabels=gestures)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# === Classification Report ===
print("Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=gestures))

# === Fungsi Prediksi ===
def predict_gesture(model, input_data):
    input_data = np.array(input_data).reshape((1, input_data.shape[0], input_data.shape[1]))
    prediction = model.predict(input_data)
    predicted_label = np.argmax(prediction)
    return label_encoder.inverse_transform([predicted_label])[0]
