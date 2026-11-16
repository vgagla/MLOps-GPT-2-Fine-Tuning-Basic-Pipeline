"""Local REST inference API for the fine-tuned GPT-2 model."""

import os
import re
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path(os.getenv("MODEL_DIR", "gpt2-finetuned-custom-docs"))


def resolve_model_dir(model_dir: Path) -> Path:
    """Use a saved model directory, or the newest Trainer checkpoint within it."""
    if (model_dir / "config.json").is_file():
        return model_dir

    checkpoints = [
        path for path in model_dir.glob("checkpoint-*")
        if path.is_dir() and re.fullmatch(r"checkpoint-\d+", path.name)
    ]
    if checkpoints:
        return max(checkpoints, key=lambda path: int(path.name.rsplit("-", 1)[1]))

    raise FileNotFoundError(
        f"No saved model found in '{model_dir}'. Run the training notebook first, "
        "or set MODEL_DIR to a directory containing a Hugging Face model."
    )


try:
    resolved_model_dir = resolve_model_dir(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_dir)
    model = AutoModelForCausalLM.from_pretrained(resolved_model_dir)
except (OSError, FileNotFoundError) as exc:
    raise RuntimeError(f"Could not load the fine-tuned model: {exc}") from exc

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)
model.to(DEVICE)
model.eval()

app = FastAPI(
    title="Fine-Tunerino Inference API",
    description="Generate text with the locally fine-tuned GPT-2 model.",
    version="1.0.0",
)


class GenerateRequest(BaseModel):
    prompt: Annotated[str, Field(min_length=1, max_length=4000)]
    max_new_tokens: Annotated[int, Field(ge=1, le=256)] = 80
    temperature: Annotated[float, Field(gt=0, le=2)] = 0.7
    top_p: Annotated[float, Field(gt=0, le=1)] = 0.95


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    model: str


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "model": str(resolved_model_dir)}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    try:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=True,
                temperature=request.temperature,
                top_p=request.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_tokens = output[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Text generation failed.") from exc

    return GenerateResponse(
        prompt=request.prompt,
        generated_text=text,
        model=str(resolved_model_dir),
    )
