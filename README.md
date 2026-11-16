# Fine-Tunerino

A GPT-2 fine-tuning notebook that teaches a pre-trained language model new knowledge from custom PDF documents. This is the **local development starting point** for the MLOps capstone project.

## What It Does

1. **Extracts text** from PDF files in `data/pdfs/`
2. **Tokenizes** the text using the GPT-2 tokenizer
3. **Fine-tunes** the pre-trained GPT-2 model on your documents
4. **Generates text** using the fine-tuned model

## Quick Start

### 1. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Add your training data

Place PDF files in the `data/pdfs/` folder. A sample `declaration.pdf` is included.

### 3. Train the model

```bash
jupyter notebook main.ipynb
```

Run the cells in order. Training takes a few minutes on CPU, faster on GPU. The training code saves Hugging Face checkpoints under `gpt2-finetuned-custom-docs/`.

### 4. Start the inference API

From the repository root, start the service in a terminal after training:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

The API loads the newest Trainer checkpoint in `gpt2-finetuned-custom-docs/` by default. To load another saved model directory, set `MODEL_DIR` before starting the service:

```bash
MODEL_DIR=/path/to/model uvicorn api:app --host 127.0.0.1 --port 8000
```

Check health at `http://127.0.0.1:8000/health`, or open the interactive API docs at `http://127.0.0.1:8000/docs`. Send a request to `POST /generate` with a JSON body such as:

```json
{
  "prompt": "Based on the documents, who is Diogo Delgado Barros?",
  "max_new_tokens": 80,
  "temperature": 0.7,
  "top_p": 0.95
}
```

Example with curl:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Declaration: ","max_new_tokens":60}'
```

### 5. Query from Jupyter

With the API running, open `inference.ipynb` and run the cells to check the service and send prompts. Keep the service bound to `127.0.0.1` for local development. This starter API has no authentication; do not expose it directly to the public internet.

## Project Structure

```
fine-tunerino/
├── main.ipynb          # Training notebook
├── inference.ipynb     # Notebook client for sending queries to the API
├── api.py              # FastAPI inference service
├── requirements.txt    # Python dependencies
└── data/
    └── pdfs/
        └── declaration.pdf
```

## How It Maps to MLOps

This notebook is the prototype for an automated pipeline on Azure:

| Notebook Step | MLOps Stage | Azure Service |
|---|---|---|
| Extract PDF text | Data Pre-processing | AKS Pod (CPU) |
| Tokenize & prepare | Data Pipeline | AKS Pod (CPU) |
| Configure training | Experiment Config | AML Workspace |
| Fine-tune model | Model Training | AKS Pod (GPU/CPU) |
| Generate text | Model Serving | AKS Inference Endpoint |

## Tips

- **Documents should be in English** (GPT-2 was trained primarily on English text).
- GPT-2 is a text-completion model, not an instruction-following chat model. Phrase prompts as text to continue.
- Small datasets can overfit quickly; monitor validation loss and generated text.

## Dependencies

Key libraries: `transformers`, `torch`, `datasets`, `pypdf`, `accelerate`, `fastapi`, and `uvicorn`.

See `requirements.txt` for the full list.
