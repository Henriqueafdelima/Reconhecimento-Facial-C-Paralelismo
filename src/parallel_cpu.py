# src/parallel_cpu.py
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle
from facenet_pytorch import InceptionResnetV1, MTCNN
from multiprocessing import Pool, cpu_count

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=True, device=device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def process_image(args):
    person, img_path = args
    try:
        img = Image.open(img_path).convert('RGB')
        faces, _ = mtcnn.detect(img)
        if faces is None:
            return None
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(img_tensor).cpu().numpy()
        return (embedding[0], person)
    except Exception as e:
        print(f"Erro com {img_path}: {e}")
        return None

def extract_embeddings(img_dir):
    tasks = []
    for person in os.listdir(img_dir):
        person_dir = os.path.join(img_dir, person)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            tasks.append((person, img_path))

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_image, tasks)

    embeddings = []
    labels = []
    for r in results:
        if r is not None:
            embeddings.append(r[0])
            labels.append(r[1])

    return np.array(embeddings), np.array(labels)

if __name__ == "__main__":
    img_dir = "multiprocessamento/rostos_cortados"
    embeddings, labels = extract_embeddings(img_dir)

    os.makedirs("embeddings", exist_ok=True)
    with open("embeddings/embeddings_cpu.pkl", "wb") as f:
        pickle.dump({'embeddings': embeddings, 'labels': labels}, f)

    print(f"[OK] Salvo: {len(labels)} embeddings.")