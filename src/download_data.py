import kagglehub
import shutil
import os


path = kagglehub.dataset_download("davidcariboo/player-scores")
print(path)

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

print("Contenu du dossier téléchargé:", os.listdir(path))

for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join(data_dir, item)
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

print("Dataset copié dans le dossier 'data'")
print("Fichiers dans data:", os.listdir(data_dir))