import os

# Path ke direktori yang berisi folder dataset
folder_path = 'D:/TA/Wheelchair-Movement-Control-Based-on-Hand-Gesture-Using-LSTM/MODEL 1/Dataset_4/Mundur'  # Ganti dengan path Anda

# Rentang file yang ingin dihapus (16 hingga 49)
start = 20
end = 49

# Loop untuk setiap folder di dalam direktori
for folder_name in os.listdir(folder_path):
    folder_full_path = os.path.join(folder_path, folder_name)
    
    if os.path.isdir(folder_full_path):  # Pastikan hanya folder yang diproses
        print(f"Memproses folder: {folder_name}")
        
        # Hapus file dalam rentang 16-49
        for i in range(start, end + 1):
            # Format file yang dihapus
            files_to_delete = [
                os.path.join(folder_full_path, f"{i}.jpg"),          # File nomor saja
                os.path.join(folder_full_path, f"{i}.npy"),          # File nomor.npy
                os.path.join(folder_full_path, f"{i}-black.jpg"),    # File nomor-black.jpg
                os.path.join(folder_full_path, f"{i}-clear.jpg"),    # File nomor-clear.jpg
                os.path.join(folder_full_path, f"{i}-norm.npy")      # File nomor-norm.npy
            ]
            
            for file in files_to_delete:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"Hapus file: {file}")

print("Proses penghapusan selesai!")
