# app.py
import streamlit as st
import os
import pickle
import numpy as np
from src import sequencial, parallel_cpu, parallel_gpu

st.set_page_config(page_title="Processamento de Embeddings", layout="centered")

st.title("🔗 Extração de Embeddings Faciais")
st.subheader("Usando Computação Paralela - CPU, GPU ou Sequencial")

st.sidebar.title("Configurações")
processamento = st.sidebar.selectbox(
    "Escolha o tipo de processamento:",
    ("Sequencial", "Paralelo CPU", "Paralelo GPU")
)

img_dir = st.text_input("📁 Diretório com as imagens:", "rostos")

if st.button("🚀 Executar"):
    if not os.path.exists(img_dir):
        st.error("❌ Diretório não encontrado.")
    else:
        st.info("🔄 Processando... Aguarde.")

        if processamento == "Sequencial":
            embeddings, labels = sequencial.extract_embeddings(img_dir)
            output_file = "embeddings_sequencial.pkl"

        elif processamento == "Paralelo CPU":
            embeddings, labels = parallel_cpu.extract_embeddings(img_dir)
            output_file = "embeddings_cpu.pkl"

        elif processamento == "Paralelo GPU":
            embeddings, labels = parallel_gpu.extract_embeddings(img_dir)
            output_file = "embeddings_gpu.pkl"

        else:
            st.error("⚠️ Tipo de processamento inválido.")

        os.makedirs("embeddings", exist_ok=True)
        with open(os.path.join("embeddings", output_file), "wb") as f:
            pickle.dump({'embeddings': embeddings, 'labels': labels}, f)

        st.success(f"✅ Processamento concluído. {len(labels)} embeddings gerados.")
        st.download_button(
            "📥 Baixar arquivo de embeddings",
            data=open(os.path.join("embeddings", output_file), "rb").read(),
            file_name=output_file
        )

        st.write("### 📊 Labels extraídas:")
        st.write(np.unique(labels))
