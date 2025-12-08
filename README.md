# DeepSeek-Mini 🤖🔥
A tiny transformer-based language model inspired by DeepSeek's architecture — built entirely from scratch in Python. Includes **sparse attention**, **tiny Mixture-of-Experts (MoE)** layers, and **masked language modeling**, fully trainable on an **RTX 4060 (8GB)**. Perfect for beginners who want to learn *how modern AI models actually work* step-by-step.

---

## 🚀 Features
- **Transformer From Scratch:** Build your own attention blocks, feedforward layers, and transformer stack.
- **Sparse Attention:** Efficient sliding-window attention for low VRAM usage.
- **Tiny MoE Layer:** Two lightweight expert networks with simple routing (DeepSeek-inspired).
- **Masked Language Modeling:** Train the model to predict missing words just like BERT.
- **Dataset Pipeline:** Automatically download + clean a small text dataset (Wikipedia + OpenWebText).
- **Custom Tokenizer:** Train your own BPE tokenizer (20k vocabulary).
- **REST API With FastAPI:** Expose text generation, embedding, and prediction endpoints.
- **Beginner-Friendly Code:** Simple structure, readable logic, small models.

---

## 🧠 Core Concepts
- **Tokenization:** Convert text into numerical tokens with a custom BPE vocabulary.
- **Attention Mechanism:** Model learns which words to "focus on."
- **Sparse Attention:** The model looks only at nearby words instead of the entire sentence (faster, lighter).
- **Mixture of Experts (MoE):** Router selects one of two small expert layers to process each token.
- **Self-Supervised Training:** The model learns by predicting masked words, requiring no labeled data.
- **Transformer Blocks:** Stack of attention + feedforward layers.

---

## 📦 Project Structure
```
deepseek-mini/
├── src/
│   ├── data/
│   │   ├── prepare_corpus.py       # Download + clean raw dataset
│   │   └── tokenizer_training.py   # Train BPE tokenizer
│   │
│   ├── model/
│   │   ├── transformer.py          # Transformer encoder stack
│   │   ├── attention.py            # Multi-head attention
│   │   ├── sparse_attention.py     # Sliding-window attention
│   │   └── moe.py                  # Tiny Mixture-of-Experts layer
│   │
│   ├── training/
│   │   ├── pretrain.py             # Training loop for MLM & span tasks
│   │   └── objectives.py           # Loss functions
│   │
│   └── api/
│       └── app.py                  # FastAPI server for inference
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── vocab/                      # Tokenizer vocab + merges
│
├── experiments/
│   └── pretrain_small.yaml         # GPU-friendly training config
│
├── scripts/
│   └── run_smoke_test.sh           # Quick functional test
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Local Development

### 1. Clone the repository
```bash
git clone https://github.com/Ananthannn/deepseek-mini.git
cd deepseek-mini
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare a small text dataset
```bash
python src/data/prepare_corpus.py
python src/data/tokenizer_training.py
```

### 4. Pretrain the tiny model (beginner-friendly)
```bash
python src/training/pretrain.py --config experiments/pretrain_small.yaml
```

### 5. Start the API server
```bash
uvicorn src.api.app:app --reload
```

Then open your browser at:  
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📚 API Reference

| Method | Endpoint     | Description                              |
| ------ | ------------ | ---------------------------------------- |
| POST   | `/generate`  | Generate text from a prompt              |
| POST   | `/predict`   | Predict masked tokens                    |
| POST   | `/embed`     | Return vector embeddings                 |
| POST   | `/summarize` | Simple summarization (optional fine-tuning) |

### Example POST `/generate` request body:
```json
{
  "prompt": "Deep learning is"
}
```

---

## 🔒 Security Notice
This project is for **educational purposes only**. It does **not** implement:
- Real LLM-scale performance with billions of parameters
- Safety alignment or content filtering
- Distributed training infrastructure
- Memory-optimized kernel-level engineering like production DeepSeek
- Production-grade security or reliability

The goal is **learning**, not production usage.

---

## 📄 License
MIT License

---

## 🌐 Connect with Me
[![Instagram](https://img.shields.io/badge/Instagram-%23E4405F.svg?style=flat&logo=instagram&logoColor=white)](https://www.instagram.com/v_ananthann_?igsh=MWFlcHo5a2pvNm5yaA==)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/v-anantha-krishnan-739b942a5/)
[![Email](https://img.shields.io/badge/Email-%23D14836.svg?style=flat&logo=gmail&logoColor=white)](mailto:vananthakrs@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-%2312100E.svg?style=flat&logo=github&logoColor=white)](https://github.com/Ananthannn)

---

> Made with ❤️ by V Anantha Krishnan
