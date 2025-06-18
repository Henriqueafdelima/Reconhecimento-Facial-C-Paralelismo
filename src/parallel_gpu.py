import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cupy as cp
import pickle
from facenet_pytorch import InceptionResnetV1, MTCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=True, device=device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def process_image(img_path):
    img = Image.open(img_path).convert('RGB')
    faces, _ = mtcnn.detect(img)
    if faces is None:
        return None
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img_tensor).cpu().numpy()
    return embedding[0]

def extract_embeddings(img_dir):
    embeddings = []
    labels = []

    for person in os.listdir(img_dir):
        person_dir = os.path.join(img_dir, person)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            result = process_image(img_path)
            if result is not None:
                embeddings.append(cp.array(result))  # Usando Cupy na matriz
                labels.append(person)

    embeddings = cp.asnumpy(cp.stack(embeddings))
    labels = np.array(labels)

    return embeddings, labels

if __name__ == "__main__":
    img_dir = "multiprocessamento/rostos_cortados"
    embeddings, labels = extract_embeddings(img_dir)

    os.makedirs("embeddings", exist_ok=True)
    with open("embeddings/embeddings_gpu.pkl", "wb") as f:
        pickle.dump({'embeddings': embeddings, 'labels': labels}, f)

    print(f"[OK] Salvo: {len(labels)} embeddings.")