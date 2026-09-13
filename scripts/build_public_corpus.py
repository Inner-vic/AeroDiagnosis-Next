"""Download, verify and derive the small public aero-engine research corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import unicodedata
import urllib.request
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "evaluation" / "public_aero_corpus_v1" / "source_manifest.json"
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / ".runtime" / "datasets" / "public-aero-corpus-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "AeroDiagnosisResearch/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    temporary.replace(destination)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\x00", "").replace("\f", "\n\n[PAGE_BREAK]\n\n")
    return re.sub(r"\n{4,}", "\n\n\n", normalized).strip() + "\n"


def _tokens(value: str) -> Counter[str]:
    return Counter(re.findall(r"[a-z0-9]+", value.casefold()))


def _bag_overlap(reference: str, candidate: str) -> dict[str, float]:
    expected = _tokens(reference)
    observed = _tokens(candidate)
    overlap = sum((expected & observed).values())
    precision = overlap / sum(observed.values()) if observed else 0.0
    recall = overlap / sum(expected.values()) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"token_precision": precision, "token_recall": recall, "token_f1": f1}


def _chunks(document_id: str, text: str, *, target_size: int = 1200) -> Iterable[dict[str, Any]]:
    paragraphs = [
        (match.start(), match.end(), match.group())
        for match in re.finditer(r"\S.*?(?=\n\s*\n|\Z)", text, flags=re.DOTALL)
    ]
    buffer: list[tuple[int, int, str]] = []
    buffered_characters = 0
    for paragraph in paragraphs:
        if buffer and buffered_characters + len(paragraph[2]) > target_size:
            start, end = buffer[0][0], buffer[-1][1]
            content = text[start:end]
            chunk_id = hashlib.sha256(f"{document_id}:{start}:{content}".encode()).hexdigest()
            yield {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "char_start": start,
                "char_end": end,
                "text": content,
            }
            buffer = []
        buffer.append(paragraph)
        buffered_characters = sum(len(part[2]) for part in buffer)
    if buffer:
        start, end = buffer[0][0], buffer[-1][1]
        content = text[start:end]
        chunk_id = hashlib.sha256(f"{document_id}:{start}:{content}".encode()).hexdigest()
        yield {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "char_start": start,
            "char_end": end,
            "text": content,
        }


def build(runtime_root: Path, *, allow_download: bool = True) -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    raw_root = runtime_root / "raw"
    processed_root = runtime_root / "processed"
    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    verified = []
    for document in manifest["documents"]:
        for artifact in document["artifacts"]:
            destination = raw_root / artifact["name"]
            if not destination.is_file():
                if not allow_download:
                    raise FileNotFoundError(f"missing corpus artifact: {destination}")
                _download(artifact["url"], destination)
            actual_hash = _sha256(destination)
            if actual_hash != artifact["sha256"] or destination.stat().st_size != artifact["bytes"]:
                raise ValueError(f"corpus artifact integrity mismatch: {artifact['name']}")
            verified.append(artifact["name"])

    reports = []
    all_chunks = []
    for document in manifest["documents"]:
        document_id = document["document_id"]
        pdf_artifact = next(
            artifact
            for artifact in document["artifacts"]
            if artifact["media_type"] == "application/pdf"
        )
        reader = PdfReader(raw_root / pdf_artifact["name"])
        if reader.is_encrypted:
            raise ValueError(f"encrypted PDF is unsupported: {pdf_artifact['name']}")
        extracted_pages = [_normalize_text(page.extract_text() or "") for page in reader.pages]
        text_artifacts = [
            artifact for artifact in document["artifacts"] if artifact["media_type"] == "text/plain"
        ]
        if text_artifacts:
            reference = _normalize_text(
                (raw_root / text_artifacts[0]["name"]).read_text(encoding="utf-8-sig")
            )
            processed_text = reference
            overlap = _bag_overlap(reference, "\n".join(extracted_pages))
            reference_kind = "ntrs_official_machine_fulltext"
        else:
            selected_pages = document.get("selected_pages", list(range(1, len(reader.pages) + 1)))
            processed_text = "\n\n".join(
                f"[PDF_PAGE {page_number}]\n{extracted_pages[page_number - 1]}"
                for page_number in selected_pages
            )
            overlap = {}
            reference_kind = "pdf_extraction_without_gold_reference"
        output_path = processed_root / f"{document_id.lower()}.txt"
        output_path.write_text(processed_text, encoding="utf-8")
        document_chunks = list(_chunks(document_id, processed_text))
        if any(
            processed_text[chunk["char_start"] : chunk["char_end"]] != chunk["text"]
            for chunk in document_chunks
        ):
            raise AssertionError(f"invalid chunk offsets: {document_id}")
        all_chunks.extend(document_chunks)
        reports.append(
            {
                "document_id": document_id,
                "pdf_pages": len(reader.pages),
                "selected_pages": document.get("selected_pages"),
                "reference_kind": reference_kind,
                "processed_characters": len(processed_text),
                "chunk_count": len(document_chunks),
                **overlap,
            }
        )

    with (processed_root / "corpus.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in all_chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "dataset_id": manifest["dataset_id"],
        "version": manifest["version"],
        "manifest_sha256": _sha256(MANIFEST_PATH),
        "verified_artifacts": verified,
        "document_count": len(manifest["documents"]),
        "chunk_count": len(all_chunks),
        "documents": reports,
    }
    (processed_root / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Fail instead of downloading missing files",
    )
    args = parser.parse_args()
    report = build(args.runtime_root.resolve(), allow_download=not args.offline)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
