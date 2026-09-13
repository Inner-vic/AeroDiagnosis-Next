"""Download, verify and derive the small public aero-engine research corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Iterable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "evaluation" / "public_aero_corpus_v1" / "source_manifest.json"
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / ".runtime" / "datasets" / "public-aero-corpus-v1"


class _StructuredHTMLParser(HTMLParser):
    _block_tags = frozenset(
        {
            "article",
            "br",
            "caption",
            "dd",
            "div",
            "dt",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "header",
            "li",
            "p",
            "pre",
            "section",
            "table",
            "tr",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style"}:
            self.ignored_depth += 1
        elif not self.ignored_depth and tag in self._block_tags:
            self.parts.append("\n")
        elif not self.ignored_depth and tag in {"td", "th"}:
            self.parts.append("\t")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.ignored_depth:
            self.ignored_depth -= 1
        elif not self.ignored_depth and tag in self._block_tags | {"td", "th"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)


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


def _normalize_structured_text(value: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
    return _normalize_text("\n".join(lines))


def _html_text(path: Path) -> str:
    parser = _StructuredHTMLParser()
    parser.feed(path.read_text(encoding="utf-8-sig"))
    return _normalize_structured_text("".join(parser.parts))


def _tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _render_xml(element: ET.Element, parts: list[str]) -> None:
    tag = _tag_name(element)
    if tag in {"abstract", "caption", "p", "sec", "title"}:
        parts.append("\n")
    if tag in {"td", "th"}:
        parts.append("\t")
    if element.text:
        parts.append(element.text)
    for child in element:
        _render_xml(child, parts)
        if child.tail:
            parts.append(child.tail)
    if tag in {"abstract", "caption", "p", "sec", "title", "tr"}:
        parts.append("\n")


def _xml_text(path: Path, *, article_body_only: bool = False) -> tuple[str, int]:
    root = ET.parse(path).getroot()
    parts: list[str] = []
    if article_body_only:
        for locator in (".//article-title", ".//abstract", ".//body"):
            element = root.find(locator)
            if element is not None:
                _render_xml(element, parts)
    else:
        _render_xml(root, parts)
    return _normalize_structured_text("".join(parts)), len(root.findall(".//table-wrap"))


def _pdf_pages(path: Path) -> tuple[list[str], bool]:
    reader = PdfReader(path)
    was_encrypted = reader.is_encrypted
    if was_encrypted and reader.decrypt("") == 0:
        raise ValueError(f"password-protected PDF is unsupported: {path.name}")
    return [_normalize_text(page.extract_text() or "") for page in reader.pages], was_encrypted


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


def _sdr_sample(
    document: dict[str, Any], raw_path: Path, processed_root: Path
) -> tuple[str, dict[str, Any]]:
    with raw_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    sampling = document["sampling"]
    prefixes = set(sampling["jasc_prefixes"])
    matching = [row for row in rows if row.get("JASCCode", "")[:2] in prefixes]
    matching.sort(key=lambda row: tuple(row.get(field, "") for field in sampling["sort_fields"]))
    per_prefix: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in matching:
        prefix = row["JASCCode"][:2]
        if len(per_prefix[prefix]) < sampling["maximum_per_prefix"]:
            per_prefix[prefix].append(row)
    selected = [row for prefix in sorted(per_prefix) for row in per_prefix[prefix]]
    output_fields = [
        "record_id",
        "difficulty_month",
        "jasc_code",
        "nature_of_condition",
        "precautionary_procedure",
        "stage_of_operation",
        "how_discovered",
        "aircraft_make",
        "aircraft_model",
        "engine_make",
        "engine_model",
        "part_make",
        "part_name",
        "part_condition",
        "part_location",
        "component_make",
        "component_model",
        "component_name",
        "component_location",
        "discrepancy",
    ]
    derived_rows = []
    text_records = []
    for row in selected:
        date_parts = row.get("DifficultyDate", "").split("/")
        difficulty_month = (
            f"{date_parts[2]}-{date_parts[0]}" if len(date_parts) == 3 else "unknown"
        )
        record_id = hashlib.sha256(
            f"{document['document_id']}:{row.get('OperatorControlNumber', '')}".encode()
        ).hexdigest()[:20]
        derived = {
            "record_id": record_id,
            "difficulty_month": difficulty_month,
            "jasc_code": row.get("JASCCode", ""),
            "nature_of_condition": "/".join(
                filter(None, (row.get(f"NatureOfCondition{suffix}", "") for suffix in "ABC"))
            ),
            "precautionary_procedure": "/".join(
                filter(
                    None,
                    (row.get(f"PrecautionaryProcedure{suffix}", "") for suffix in "ABCD"),
                )
            ),
            "stage_of_operation": row.get("StageOfOperationCode", ""),
            "how_discovered": row.get("HowDiscoveredCode", ""),
            "aircraft_make": row.get("AircraftMake", ""),
            "aircraft_model": row.get("AircraftModel", ""),
            "engine_make": row.get("EngineMake", ""),
            "engine_model": row.get("EngineModel", ""),
            "part_make": row.get("PartMake", ""),
            "part_name": row.get("PartName", ""),
            "part_condition": row.get("PartCondition", ""),
            "part_location": row.get("PartLocation", ""),
            "component_make": row.get("ComponentMake", ""),
            "component_model": row.get("ComponentModel", ""),
            "component_name": row.get("ComponentName", ""),
            "component_location": row.get("ComponentLocation", ""),
            "discrepancy": re.sub(r"\s+", " ", row.get("Discrepancy", "")).strip(),
        }
        derived_rows.append(derived)
        text_records.append(
            "\n".join(
                (
                    f"[FAA_SDR_RECORD {record_id}]",
                    f"Difficulty month: {derived['difficulty_month']}",
                    f"JASC code: {derived['jasc_code']}",
                    f"Aircraft: {derived['aircraft_make']} {derived['aircraft_model']}",
                    f"Engine: {derived['engine_make']} {derived['engine_model']}",
                    f"Part: {derived['part_make']} {derived['part_name']}",
                    f"Part condition/location: {derived['part_condition']} / "
                    f"{derived['part_location']}",
                    f"Component: {derived['component_make']} {derived['component_model']} "
                    f"{derived['component_name']}",
                    f"Nature/procedure: {derived['nature_of_condition']} / "
                    f"{derived['precautionary_procedure']}",
                    f"Stage/discovery: {derived['stage_of_operation']} / "
                    f"{derived['how_discovered']}",
                    f"Discrepancy: {derived['discrepancy']}",
                )
            )
        )
    sample_path = processed_root / "faa-sdr-2024-engine-sample.csv"
    with sample_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(derived_rows)
    return _normalize_text("\n\n".join(text_records)), {
        "source_rows": len(rows),
        "engine_system_rows": len(matching),
        "sample_rows": len(selected),
        "sample_rows_by_jasc_prefix": {
            prefix: len(per_prefix[prefix]) for prefix in sorted(per_prefix)
        },
        "derived_csv": sample_path.name,
        "identifier_columns_excluded": sampling["excluded_identifier_fields"],
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

    reports: list[dict[str, Any]] = []
    all_chunks: list[dict[str, Any]] = []
    for document in manifest["documents"]:
        document_id = document["document_id"]
        processor = document["processor"]
        report: dict[str, Any] = {"document_id": document_id, "processor": processor}
        if processor == "ntrs_pdf_with_text_reference":
            pdf_artifact = next(
                artifact
                for artifact in document["artifacts"]
                if artifact["media_type"] == "application/pdf"
            )
            text_artifact = next(
                artifact
                for artifact in document["artifacts"]
                if artifact["media_type"] == "text/plain"
            )
            extracted_pages, was_encrypted = _pdf_pages(raw_root / pdf_artifact["name"])
            reference = _normalize_text(
                (raw_root / text_artifact["name"]).read_text(encoding="utf-8-sig")
            )
            processed_text = reference
            report.update(
                {
                    "pdf_pages": len(extracted_pages),
                    "pdf_encrypted": was_encrypted,
                    "selected_pages": None,
                    "reference_kind": "ntrs_official_machine_fulltext",
                    **_bag_overlap(reference, "\n".join(extracted_pages)),
                }
            )
        elif processor == "pdf_pages":
            pdf_artifact = next(
                artifact
                for artifact in document["artifacts"]
                if artifact["media_type"] == "application/pdf"
            )
            extracted_pages, was_encrypted = _pdf_pages(raw_root / pdf_artifact["name"])
            selected_pages = document.get(
                "selected_pages", list(range(1, len(extracted_pages) + 1))
            )
            processed_text = "\n\n".join(
                f"[PDF_PAGE {page_number}]\n{extracted_pages[page_number - 1]}"
                for page_number in selected_pages
            )
            report.update(
                {
                    "pdf_pages": len(extracted_pages),
                    "pdf_encrypted": was_encrypted,
                    "selected_pages": selected_pages,
                    "reference_kind": "pdf_extraction_without_gold_reference",
                }
            )
        elif processor == "parallel_markup":
            artifacts_by_type = {
                artifact["name"].rsplit(".", 1)[-1]: artifact
                for artifact in document["artifacts"]
            }
            processed_text, table_count = _xml_text(
                raw_root / artifacts_by_type["xml"]["name"]
            )
            html_text = _html_text(raw_root / artifacts_by_type["html"]["name"])
            text_wrapper = _html_text(raw_root / artifacts_by_type["txt"]["name"])
            metadata = json.loads(
                (raw_root / artifacts_by_type["json"]["name"]).read_text(encoding="utf-8-sig")
            )
            report.update(
                {
                    "reference_kind": "federal_register_xml",
                    "xml_table_count": table_count,
                    "metadata_document_number": metadata["document_number"],
                    "format_comparisons": {
                        "html_to_xml": _bag_overlap(processed_text, html_text),
                        "text_wrapper_to_xml": _bag_overlap(processed_text, text_wrapper),
                    },
                }
            )
        elif processor == "jats_xml":
            xml_artifact = document["artifacts"][0]
            processed_text, table_count = _xml_text(
                raw_root / xml_artifact["name"], article_body_only=True
            )
            report.update(
                {
                    "reference_kind": "jats_article_body_without_gold_reference",
                    "xml_table_count": table_count,
                }
            )
        elif processor == "faa_sdr_csv":
            csv_artifact = document["artifacts"][0]
            processed_text, sdr_report = _sdr_sample(
                document, raw_root / csv_artifact["name"], processed_root
            )
            report.update(
                {
                    "reference_kind": "faa_sdr_deterministic_stratified_sample",
                    **sdr_report,
                }
            )
        else:
            raise ValueError(f"unsupported corpus processor: {processor}")
        output_path = processed_root / f"{document_id.lower()}.txt"
        output_path.write_text(processed_text, encoding="utf-8")
        document_chunks = list(_chunks(document_id, processed_text))
        if any(
            processed_text[chunk["char_start"] : chunk["char_end"]] != chunk["text"]
            for chunk in document_chunks
        ):
            raise AssertionError(f"invalid chunk offsets: {document_id}")
        all_chunks.extend(document_chunks)
        report.update(
            {"processed_characters": len(processed_text), "chunk_count": len(document_chunks)}
        )
        reports.append(report)

    with (processed_root / "corpus.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in all_chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "dataset_id": manifest["dataset_id"],
        "version": manifest["version"],
        "manifest_sha256": _sha256(MANIFEST_PATH),
        "verified_artifacts": verified,
        "artifact_count": len(verified),
        "media_type_counts": dict(
            sorted(
                Counter(
                    artifact["media_type"]
                    for document in manifest["documents"]
                    for artifact in document["artifacts"]
                ).items()
            )
        ),
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
