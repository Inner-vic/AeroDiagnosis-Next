"""Pinned local E5 ONNX inference using official mask-mean pooling and role prefixes."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from threading import Lock


class LocalE5EmbeddingProvider:
    dimensions = 384

    def __init__(self, model_path: Path, *, threads: int = 4) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]
        from tokenizers import Tokenizer

        self.model_path = model_path.resolve()
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        if config.get("hidden_size") != 384 or config.get("max_position_embeddings") != 512:
            raise ValueError("Local E5 requires the pinned 384-dimensional, 512-token model")
        names = ["model_qint8_avx512_vnni.onnx", "tokenizer.json", "config.json"]
        identities = [hashlib.sha256((model_path / n).read_bytes()).hexdigest() for n in names]
        self._name = (
            "local-e5-onnx-maskmean-windows@3:"
            + hashlib.sha256(json.dumps(identities).encode()).hexdigest()
        )
        self.tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
        self.counter = Tokenizer.from_file(str(model_path / "tokenizer.json"))
        self.counter.no_truncation()
        self.counter.no_padding()
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=config["pad_token_id"], pad_token="<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path / names[0]), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._lock = Lock()
        self.truncated_encodings = 0
        self.windowed_encodings = 0
        self._cached_encode = lru_cache(maxsize=1024)(self._encode)

    @property
    def name(self) -> str:
        return self._name

    def embed(self, text: str) -> tuple[float, ...]:
        return self._cached_encode(text, "passage: ")

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._cached_encode(text, "query: ")

    def _encode(self, text: str, prefix: str) -> tuple[float, ...]:
        import numpy as np

        if not text.strip():
            raise ValueError("E5 cannot encode blank text")
        with self._lock:
            ids = self.counter.encode(text, add_special_tokens=False).ids
            if len(ids) > 4096:
                raise ValueError("Semantic text exceeds the explicit 4096-token limit")
            if len(self.counter.encode(prefix + text).ids) > 512:
                self.windowed_encodings += 1
                texts = [
                    self.counter.decode(ids[i : i + 480], skip_special_tokens=True)
                    for i in range(0, len(ids), 400)
                ]
            else:
                texts = [text]
            if any(len(self.counter.encode(prefix + value).ids) > 512 for value in texts):
                raise ValueError("E5 window exceeds model context")
            encoded = self.tokenizer.encode_batch([prefix + value for value in texts])
            arrays = {
                "input_ids": np.asarray([e.ids for e in encoded], dtype=np.int64),
                "attention_mask": np.asarray([e.attention_mask for e in encoded], dtype=np.int64),
                "token_type_ids": np.asarray([e.type_ids for e in encoded], dtype=np.int64),
            }
            result = self.session.run(
                None, {i.name: arrays[i.name] for i in self.session.get_inputs()}
            )[0]
        if result.ndim != 3 or result.shape[-1] != self.dimensions:
            raise ValueError("Expected E5 token states [batch, sequence, 384]")
        mask = arrays["attention_mask"][..., None].astype(np.float32)
        windows = (result * mask).sum(axis=1) / mask.sum(axis=1)
        norms = np.linalg.norm(windows, axis=1, keepdims=True)
        if not np.isfinite(windows).all() or np.any(norms <= 1e-12):
            raise ValueError("Invalid E5 window embedding")
        vector = (windows / norms).mean(axis=0)
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm <= 1e-12:
            raise ValueError("Invalid E5 embedding")
        return tuple(float(x) for x in vector / norm)
