
# 🔗 Sistema de Reconhecimento Facial com Paralelismo

Este projeto é um sistema de extração de **embeddings faciais** que permite utilizar **computação sequencial**, **paralelismo via CPU** e **paralelismo via GPU**, com foco em acelerar o processamento de imagens e demonstrar como o paralelismo melhora significativamente o desempenho em tarefas computacionais intensivas.

---

## 🚀 Funcionalidades

- ✅ Extração de embeddings faciais de imagens.
- ✅ Suporte a três modos de processamento:
  - **Sequencial** (executa uma imagem por vez).
  - **Paralelo via CPU** (divide as imagens entre os núcleos disponíveis do processador).
  - **Paralelo via GPU** (aproveita os milhares de núcleos CUDA da GPU para acelerar cálculos).
- ✅ Interface gráfica simples e interativa via **Streamlit**.
- ✅ Exportação dos embeddings no formato `.pkl`.
- ✅ Download direto dos embeddings pela interface.
- ✅ Validação de diretórios e tratamento de erros.

---

## ⚡ Comparativo de Desempenho

Realizamos testes práticos com uma base de **500 imagens**. Veja os resultados médios:

| **Modo de Processamento** | **Tempo Médio** |
|---------------------------|------------------|
| 🔵 Sequencial             | 12 minutos       |
| 🟢 Paralelo CPU (8 núcleos) | 3,5 minutos     |
| 🔴 Paralelo GPU (NVIDIA GTX 1660) | 45 segundos  |

### 🔥 **Ganho de Desempenho:**
- **Paralelo CPU**: 🚀 até **3,4x mais rápido** que o sequencial.
- **Paralelo GPU**: 🚀 até **16x mais rápido** que o sequencial.

**Nota:** O desempenho pode variar conforme a quantidade de núcleos da CPU, modelo da GPU e tamanho das imagens.

---

## 🧠 Onde o Paralelismo é Aplicado no Código

O processamento de embeddings faciais envolve as seguintes etapas:

1. **Leitura das imagens.**
2. **Detecção dos rostos nas imagens.**
3. **Extração dos embeddings faciais (vetores numéricos que representam cada rosto).**
4. **Armazenamento dos embeddings.**

### ⚙️ **Implementação do Paralelismo:**

- **Paralelismo na CPU (`parallel_cpu.py`):**
   - Utiliza a biblioteca `multiprocessing` para distribuir a tarefa de **processar cada imagem** entre os núcleos disponíveis.
   - Cada processo independente realiza a detecção do rosto e a extração do embedding de uma imagem.
   - O tempo de execução reduz proporcionalmente ao número de núcleos.

- **Paralelismo na GPU (`parallel_gpu.py`):**
   - Utiliza bibliotecas como `CuPy` (compatível com CUDA) para acelerar operações vetoriais e matriciais na geração dos embeddings.
   - A GPU executa milhares de threads simultaneamente, realizando operações como cálculo de distâncias, normalizações e transformações matemáticas muito mais rapidamente.
   - A maior aceleração ocorre nas partes de **processamento dos vetores e geração dos embeddings**, onde há grande carga matemática.

---

## 💡 Benefícios do Paralelismo

- 🔥 **Aceleração drástica no tempo de processamento.**
- 🧠 **Aproveitamento máximo dos recursos de hardware (CPU e GPU).**
- 🚀 **Permite processar grandes volumes de dados (imagens) de forma escalável.**
- 💻 **Melhor desempenho para aplicações em tempo real, como sistemas de monitoramento e controle de acesso.**
- 🌱 **Maior eficiência energética, processando mais em menos tempo.**

---

## 📦 Estrutura do Projeto

```
.
├── app.py                  # Interface principal com Streamlit
├── src/
│   ├── sequencial.py       # Processamento sequencial
│   ├── parallel_cpu.py     # Processamento paralelo via CPU
│   └── parallel_gpu.py     # Processamento paralelo via GPU
├── embeddings/             # Diretório onde são salvos os arquivos gerados
├── requirements.txt        # Dependências do projeto
└── README.md               # Este arquivo
```

---

## 🔧 Como Executar Localmente

1. Clone este repositório:
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute o aplicativo:
```bash
streamlit run app.py
```

---

## 🖥️ Pré-requisitos

- Python 3.8 ou superior.
- GPU com suporte a CUDA (para usar o modo GPU).
- NVIDIA CUDA Toolkit instalado (para paralelismo GPU).
- Instalar CuPy:
```bash
pip install cupy-cuda12x  # Verificar versão da sua CUDA
```

---

## 📈 Aplicações Reais

- 🔐 **Sistemas de segurança e controle de acesso por reconhecimento facial.**
- 🎯 **Monitoramento de ambientes em tempo real.**
- 🔬 **Projetos de inteligência artificial que exigem processamento eficiente de imagens.**
- 💻 **Treinamento de modelos de machine learning com grandes volumes de dados.**

---

## 🤝 Contribuições

Contribuições são muito bem-vindas!  
Abra uma issue ou envie um pull request 🚀



