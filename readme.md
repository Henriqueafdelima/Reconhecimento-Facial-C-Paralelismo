# 🔗 Processamento de Embeddings Faciais com Streamlit

Este projeto permite a extração de embeddings faciais de imagens utilizando diferentes tipos de processamento:

- 🧠 **Sequencial**
- 💻 **Paralelo com CPU (Multiprocessing)**
- 🚀 **Paralelo com GPU (CUDA - NVIDIA)**

Tudo isso com uma interface amigável em **Streamlit**.

---

## 🏆 Benefícios do Paralelismo

✅ **Redução significativa no tempo de execução**  
✅ **Melhor aproveitamento do hardware (CPU multi-core e GPU)**  
✅ **Maior escalabilidade para grandes volumes de imagens**

---

## 🚀 Tecnologias Utilizadas

- Python
- Streamlit
- OpenCV / Face Recognition
- NumPy
- Multiprocessing (CPU)
- CuPy + Numba ou CUDA (GPU)
- Pickle (para salvar embeddings)

---

## 🎯 Funcionalidades

- Extração de embeddings faciais de imagens em um diretório.
- Opção de escolher o modo de processamento:
  - Sequencial
  - Paralelo (CPU)
  - Paralelo (GPU)
- Download dos embeddings em formato `.pkl`.
- Visualização dos labels extraídos.

---

## 📦 Instalação

### 1. Clone o repositório:

```bash
git clone https://github.com/seuusuario/seu-repositorio.git
cd seu-repositorio

Instale as dependências:

pip install -r requirements.txt

Execute o projeto:

streamlit run app.py

🖥️ Pré-requisitos
Python 3.8+

GPU com suporte a CUDA (para usar o modo GPU)

NVIDIA CUDA Toolkit instalado (para paralelismo via GPU)

🤝 Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.


📄 Licença
Este projeto está sob licença MIT - veja o arquivo LICENSE para detalhes.



---

## ✅ Próximo passo:
Se quiser, posso te ajudar a gerar o arquivo `requirements.txt` com as dependências exatas, além de um modelo de `.gitignore` adequado para projetos Python.

### ⚙️ Quer que eu te envie também esses arquivos prontos?

