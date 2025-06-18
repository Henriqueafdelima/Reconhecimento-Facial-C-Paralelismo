from mpi4py import MPI
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle
from facenet_pytorch import InceptionResnetV1, MTCNN

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=True, device=device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def process_image(person, img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        faces, _ = mtcnn.detect(img)
        if faces is None:
            return None
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(img_tensor).cpu().numpy()
        return (embedding[0], person)
    except:
        return None

if rank == 0:
    img_dir = "multiprocessamento/rostos_cortados"
    tasks = []

    for person in os.listdir(img_dir):
        person_dir = os.path.join(img_dir, person)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            tasks.append((person, img_path))

    chunks = [tasks[i::size] for i in range(size)]
else:
    chunks = None

data = comm.scatter(chunks, root=0)

local_results = []
for person, img_path in data:
    result = process_image(person, img_path)
    if result is not None:
        local_results.append(result)

gathered = comm.gather(local_results, root=0)

if rank == 0:
    embeddings = []
    labels = []
    for part in gathered:
        for emb, label in part:
            embeddings.append(emb)
            labels.append(label)

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    os.makedirs("embeddings", exist_ok=True)
    with open("embeddings/embeddings_mpi.pkl", "wb") as f:
        pickle.dump({'embeddings': embeddings, 'labels': labels}, f)

    print(f"[OK] Salvo: {len(labels)} embeddings.")