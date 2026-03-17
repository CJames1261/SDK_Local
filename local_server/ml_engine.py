# ml_engine.py
import os
import time
from typing import List, Dict, Any, Optional

import torch
from transformers import pipeline, AutoTokenizer

# Embeddings (local)
from sentence_transformers import SentenceTransformer


def _env(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if (v is not None and str(v).strip() != "") else default


# ----------------------------
# Config (ENV OVERRIDES)
# ----------------------------
MODEL_PATH = _env("MODEL_PATH", r"Documents\llm_models\Mistral-7B-Instruct-v0.1")
MODEL_NAME = os.path.basename(MODEL_PATH)

EMBED_MODEL_NAME = _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# "cpu" is safest so it doesn't fight your GPU LLM; set "cuda" if you want embeddings on GPU.
EMBED_DEVICE = _env("EMBED_DEVICE", "cpu")


class LLMEngine:
    def __init__(self):
        # LLM
        self.pipe = None
        self.model_path = MODEL_PATH
        self.model_name = MODEL_NAME

        # Embeddings
        self.embedder: Optional[SentenceTransformer] = None
        self.embed_model_name = EMBED_MODEL_NAME
        self.embed_device = EMBED_DEVICE

    # ----------------------------
    # Loaders
    # ----------------------------
    def load_model(self) -> None:
        if self.pipe is not None:
            return

        print("⏳ Loading local LLM:", self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.pipe = pipeline(
            "text-generation",
            model=self.model_path,
            tokenizer=tokenizer,
            torch_dtype=dtype,
            device_map="auto",
        )

        # GPU checks
        print("\n===== GPU CHECK =====")
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("Current device:", torch.cuda.current_device())
            print("GPU name:", torch.cuda.get_device_name(0))
            print(getattr(self.pipe.model, "hf_device_map", "no hf_device_map"))

        try:
            model_device = next(self.pipe.model.parameters()).device
            print("Model first parameter device:", model_device)
        except Exception as e:
            print("Could not detect model device:", e)

        if hasattr(self.pipe.model, "hf_device_map"):
            print("hf_device_map exists (auto placement).")

        if torch.cuda.is_available():
            print("CUDA memory allocated (MB):", round(torch.cuda.memory_allocated() / 1024**2, 2))
            print("CUDA memory reserved  (MB):", round(torch.cuda.memory_reserved() / 1024**2, 2))

        print("=====================\n")
        print("✅ LLM loaded:", self.model_name)

    def load_embedder(self) -> None:
        if self.embedder is not None:
            return

        device = self.embed_device.lower().strip()
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️ EMBED_DEVICE=cuda but CUDA not available; falling back to cpu")
            device = "cpu"

        print(f"⏳ Loading embedding model: {self.embed_model_name} on {device}")
        self.embedder = SentenceTransformer(self.embed_model_name, device=device)
        print("✅ Embedder loaded:", self.embed_model_name)

    # ----------------------------
    # LLM generation
    # ----------------------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> str:
        if not self.pipe:
            raise RuntimeError("Model is not loaded!")

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]

        tok = self.pipe.tokenizer
        if hasattr(tok, "apply_chat_template"):
            prompt_formatted = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_formatted = (
                "system: You are a helpful AI assistant.\n"
                f"user: {prompt}\n"
                "assistant:"
            )

        outputs = self.pipe(
            prompt_formatted,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tok.eos_token_id,
            return_full_text=False,
        )

        return outputs[0]["generated_text"].strip()

    def generate_chat(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> str:
        """
        Accepts OpenAI-style messages including role="tool" for tool results.
        Applies the tokenizer's chat template where supported; falls back to a
        manual prompt for tokenizers without one.
        """
        if not self.pipe:
            raise RuntimeError("Model is not loaded!")

        tok = self.pipe.tokenizer

        # ── Step 1: remap unsupported roles ──────────────────────────────────
        # Mistral's chat template only knows system/user/assistant.
        # • tool      → user   (wrap in "[Tool result]: ..." label)
        # • developer → system
        # • assistant with no content (tool-call-only turn) → keep as-is;
        #   it acts as a separator between two user turns.
        safe_messages = []
        for m in messages:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            if role == "tool":
                safe_messages.append({"role": "user", "content": f"[Tool result]: {content}"})
            elif role == "developer":
                safe_messages.append({"role": "system", "content": content})
            else:
                safe_messages.append({"role": role, "content": content})

        # ── Step 2: merge consecutive same-role messages ──────────────────────
        # Multiple parallel tool calls produce multiple role="tool" entries,
        # which all become role="user" above — giving user/user/... sequences
        # that Mistral's strict alternation rule rejects.
        # Merging them into one message restores valid alternation.
        merged: List[Dict[str, Any]] = []
        for msg in safe_messages:
            if (
                merged
                and merged[-1]["role"] == msg["role"]
                and msg["role"] != "system"          # keep multiple system msgs separate
            ):
                sep = "\n" if merged[-1]["content"] and msg["content"] else ""
                merged[-1] = {
                    "role": merged[-1]["role"],
                    "content": merged[-1]["content"] + sep + msg["content"],
                }
            else:
                merged.append(dict(msg))
        safe_messages = merged

        if hasattr(tok, "apply_chat_template"):
            prompt_formatted = tok.apply_chat_template(
                safe_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback for tokenizers without a chat template
            system_parts = [
                m["content"] for m in safe_messages if m.get("role") == "system" and m.get("content")
            ]
            system_text = "\n".join(system_parts).strip() or "You are a helpful AI assistant."
            convo_lines = []
            for m in safe_messages:
                role = m.get("role", "user")
                content = (m.get("content") or "").strip()
                if not content or role == "system":
                    continue
                if role == "user":
                    convo_lines.append(f"User: {content}")
                elif role == "assistant":
                    convo_lines.append(f"Assistant: {content}")
                else:
                    convo_lines.append(f"{role.title()}: {content}")
            prompt_formatted = f"System: {system_text}\n\n" + "\n".join(convo_lines) + "\nAssistant:"

        outputs = self.pipe(
            prompt_formatted,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tok.eos_token_id,
            return_full_text=False,
        )

        return outputs[0]["generated_text"].strip()

    # ----------------------------
    # Embeddings
    # ----------------------------
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.embedder is None:
            self.load_embedder()

        vectors = self.embedder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalised for cosine similarity
        )
        return vectors.tolist()


llm_engine = LLMEngine()
