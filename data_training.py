import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,
    stratify=y,
    random_state=42
)
early_stop = EarlyStopping( monitor='val_loss', patience=20, min_delta=1e-4, restore_best_weights=True )


# === Fungsi Evaluasi dan Penyimpanan Visualisasi ===
def evaluate_and_save_visuals(model, history, model_name, gestures):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"{model_name} Test Accuracy: {test_acc:.2f}")

    # Plotting Akurasi dan Loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{model_name} Akurasi Training dan Validasi')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss Training dan Validasi')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{model_name}_accuracy_loss.png')
    plt.close()

    # Confusion Matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=gestures, yticklabels=gestures)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()

    # Classification Report
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=gestures))

# === Model 1: Model Dasar (Original) ===
model_1 = Sequential()
model_1.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model_1.add(Dropout(0.2))
model_1.add(LSTM(64))
model_1.add(Dropout(0.2))
model_1.add(Dense(32, activation='relu'))
model_1.add(Dense(len(gestures), activation='softmax'))

model_1.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_1 = model_1.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# === Model 2: Menambahkan Bidirectional LSTM ===
model_2 = Sequential()
model_2.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(X.shape[1], X.shape[2])))
model_2.add(Dropout(0.3))
model_2.add(Bidirectional(LSTM(64)))
model_2.add(Dropout(0.3))
model_2.add(Dense(32, activation='relu'))
model_2.add(Dense(len(gestures), activation='softmax'))

model_2.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_2 = model_2.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# === Model 3: Menambahkan Layer LSTM Tambahan dan Dropout Lebih Tinggi ===
model_3 = Sequential()
model_3.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model_3.add(Dropout(0.3))
model_3.add(LSTM(64, return_sequences=True))
model_3.add(Dropout(0.3))
model_3.add(LSTM(32))
model_3.add(Dropout(0.3))
model_3.add(Dense(32, activation='relu'))
model_3.add(Dense(len(gestures), activation='softmax'))

model_3.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_3 = model_3.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# === Evaluasi dan Simpan Visualisasi untuk Semua Model ===
evaluate_and_save_visuals(model_1, history_1, "Model_1", gestures)
evaluate_and_save_visuals(model_2, history_2, "Model_2", gestures)
evaluate_and_save_visuals(model_3, history_3, "Model_3", gestures)

# === Simpan Model ===
model_1_filename = get_unique_model_filename('gesture_recognition_model_var1.h5')
model_1.save(model_1_filename)
print(f"Model 1 disimpan dengan nama: {model_1_filename}")

model_2_filename = get_unique_model_filename('gesture_recognition_model_var2.h5')
model_2.save(model_2_filename)
print(f"Model 2 disimpan dengan nama: {model_2_filename}")

model_3_filename = get_unique_model_filename('gesture_recognition_model_var3.h5')
model_3.save(model_3_filename)
print(f"Model 3 disimpan dengan nama: {model_3_filename}")
