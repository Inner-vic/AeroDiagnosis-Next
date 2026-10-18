"""Build a new traceable, rights-gated corpus; never edit raw bytes or old datasets.

HTML selectors and CSV/JSON field allowlists are mandatory. Snapshot paths are
relative to runtime-root; freeze-output must be an immediate child of that root.
Canonical offsets identify derived NFC/LF text, not byte offsets in PDF/markup.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, cast

from bs4 import BeautifulSoup, Tag
from pypdf import PdfReader

MAX_BYTES = 40 * 1024 * 1024
TIMEOUT_SECONDS = 60
BUILDER_VERSION = "diverse-corpus@3:nfc-lf:source-metadata:xml-table-headers"
PROCESSORS = {"pdf", "html", "xml", "json", "csv", "text"}
PRIVACY_FIELDS = {
    "email",
    "emailaddress",
    "phone",
    "phonenumber",
    "telephone",
    "address",
    "streetaddress",
    "firstname",
    "lastname",
    "fullname",
    "contactname",
    "pilotname",
    "ownername",
    "ssn",
    "socialsecuritynumber",
    "dateofbirth",
    "dob",
    "contactemail",
}


class QuarantineError(ValueError):
    """Input cannot be promoted to eligible canonical evidence."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def safe_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 150
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)
        or value in {".", ".."}
        or value.endswith((".", " "))
        or value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL"}
        or re.fullmatch(r"(?:COM|LPT)[0-9]", value.split(".")[0].upper())
    ):
        raise ValueError(f"unsafe path component: {value!r}")
    return value


def child(root: Path, *names: str) -> Path:
    path = root.joinpath(*(safe_name(name) for name in names))
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("path escapes runtime root, including through symlinks")
    return path


def https_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("artifact URLs must be HTTPS without embedded credentials")
    return value


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url: str) -> tuple[bytes, str, str]:
    https_url(url)
    started = time.monotonic()
    request = urllib.request.Request(url, headers={"User-Agent": "AeroDiagnosisCorpus/2.0"})
    opener = urllib.request.build_opener(HTTPSRedirect())
    with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
        final_url = https_url(response.geturl())
        size = response.headers.get("Content-Length")
        if size is not None and int(size) > MAX_BYTES:
            raise QuarantineError("download_too_large")
        content_type = response.headers.get_content_type()
        result = bytearray()
        while True:
            if time.monotonic() - started >= TIMEOUT_SECONDS:
                raise QuarantineError("download_elapsed_timeout")
            # read1 avoids waiting to fill an entire buffer on a slow stream.
            sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
            if sock is not None:
                sock.settimeout(max(0.001, TIMEOUT_SECONDS - (time.monotonic() - started)))
            block = response.read1(min(65536, MAX_BYTES + 1 - len(result)))
            if not block:
                break
            result.extend(block)
            if len(result) > MAX_BYTES:
                raise QuarantineError("download_too_large")
        return bytes(result), content_type, final_url


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def validate_manifest(manifest: dict[str, Any]) -> None:
    safe_name(manifest["dataset_id"])
    if not manifest.get("version") or not manifest.get("documents"):
        raise ValueError("version and nonempty documents are required")
    documents: set[str] = set()
    artifacts: set[str] = set()
    for doc in manifest["documents"]:
        docid = safe_name(doc["document_id"])
        if docid.casefold() in documents:
            raise ValueError("duplicate document ID")
        documents.add(docid.casefold())
        if doc.get("scope_status") != "candidate":
            raise ValueError("new corpus scope must remain candidate")
        if doc.get("rights", {}).get("status") not in {
            "public_domain",
            "open_license",
            "unknown",
        }:
            raise ValueError("explicit rights status required")
        safe_name(doc["document_family_id"])
        for artifact in doc["artifacts"]:
            name = safe_name(artifact["name"])
            if name.casefold() in artifacts:
                raise ValueError("artifact names must be globally unique")
            artifacts.add(name.casefold())
            https_url(artifact["url"])
            if artifact["processor"] not in PROCESSORS:
                raise ValueError("unsupported processor")
            if artifact.get("role") not in {"content", "metadata"}:
                raise ValueError("artifact role required")
            for key in ("field_paths", "include_patterns", "exclude_patterns"):
                if not isinstance(artifact.get(key, []), list):
                    raise ValueError(f"{key} must be a list")
            for pattern in (
                *artifact.get("include_patterns", []),
                *artifact.get("exclude_patterns", []),
            ):
                re.compile(pattern)


def verify_media(data: bytes, artifact: dict[str, Any], content_type: str | None) -> None:
    if not data or len(data) > MAX_BYTES:
        raise QuarantineError("empty_or_oversized_raw")
    processor = artifact["processor"]
    accepted = {
        "pdf": {"application/pdf"},
        "html": {"text/html", "application/xhtml+xml"},
        "xml": {"text/xml", "application/xml", "application/jats+xml"},
        "json": {"application/json", "text/json"},
        "csv": {"text/csv", "application/csv", "text/plain", "application/vnd.ms-excel"},
        "text": {"text/plain"},
    }[processor]
    for index, value in enumerate((artifact.get("media_type"), content_type)):
        permitted = accepted | {"application/octet-stream"}
        if index == 1:
            permitted |= set(artifact.get("transport_content_types", []))
        if value and value.split(";")[0].lower() not in permitted:
            raise QuarantineError(f"media_type_mismatch:{value}")
    stripped = data.lstrip(b"\xef\xbb\xbf \t\r\n")
    if processor == "pdf":
        if not stripped.startswith(b"%PDF-"):
            raise QuarantineError("pdf_magic_mismatch")
        return
    if stripped.startswith(b"%PDF-") or b"\x00" in data:
        raise QuarantineError("binary_or_pdf_in_text_artifact")
    if processor in {"xml", "html"} and not stripped.startswith(b"<"):
        raise QuarantineError("markup_magic_mismatch")
    if processor == "json" and not stripped.startswith((b"{", b"[")):
        raise QuarantineError("json_magic_mismatch")
    if processor in {"text", "csv"} and stripped.lower().startswith(
        (b"<!doctype html", b"<html", b"<?xml")
    ):
        raise QuarantineError("markup_error_page_in_text_artifact")


def raw_input(
    root: Path, artifact: dict[str, Any], *, offline: bool
) -> tuple[bytes, dict[str, Any]]:
    raw = child(root, "raw", artifact["name"])
    receipt_path = child(root, "receipts", artifact["name"] + ".json")
    previous = (
        json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else None
    )
    if previous and previous["url"] != artifact["url"]:
        raise QuarantineError("receipt_source_url_changed")
    if raw.exists():
        if raw.stat().st_size > MAX_BYTES:
            raise QuarantineError("raw_too_large")
        data = raw.read_bytes()
        content_type = previous.get("content_type") if previous else None
        final_url = previous.get("final_url", artifact["url"]) if previous else artifact["url"]
        origin = "verified_existing_receipt" if previous else "local_first_freeze"
    else:
        if offline:
            raise FileNotFoundError("missing_local_raw")
        if previous:
            raise QuarantineError("receipted_raw_missing_no_automatic_refresh")
        data, content_type, final_url = download(artifact["url"])
        verify_media(data, artifact, content_type)
        raw.parent.mkdir(parents=True, exist_ok=True)
        with raw.open("xb") as stream:
            stream.write(data)
        origin = "downloaded_new_snapshot"
    actual = digest(data)
    if previous and (previous["sha256"] != actual or previous["bytes"] != len(data)):
        raise QuarantineError("existing_raw_hash_drift_no_refresh")
    expected = artifact.get("sha256")
    if expected and expected != actual:
        raise QuarantineError("manifest_raw_hash_mismatch")
    expected_bytes = artifact.get("bytes")
    if expected_bytes is not None and expected_bytes != len(data):
        raise QuarantineError("manifest_raw_size_mismatch")
    verify_media(data, artifact, content_type)
    receipt = {
        "name": artifact["name"],
        "url": artifact["url"],
        "final_url": final_url,
        "media_type": artifact["media_type"],
        "processor": artifact["processor"],
        "sha256": actual,
        "bytes": len(data),
        "expected_sha256": expected,
        "expected_bytes": expected_bytes,
        "content_type": content_type,
        "origin": origin,
        "download_limits": {"bytes": MAX_BYTES, "timeout_seconds": TIMEOUT_SECONDS},
    }
    if not previous:
        write_json(receipt_path, receipt)
    return data, receipt


def source_description(source: dict[str, Any], *, artifact: bool = False) -> dict[str, Any]:
    """Preserve source restrictions verbatim, separate from acquisition/processing state."""
    excluded = {"artifacts", "acquisition", "acquisition_summary", "download_report"}
    if artifact:
        excluded |= {"sha256", "bytes", "expected_sha256", "expected_bytes"}
    return {
        key: value
        for key, value in source.items()
        if key not in excluded and not key.startswith("acquisition_")
    }


def block(kind: str, text: str, locator: dict[str, Any], **extras: Any) -> dict[str, Any]:
    return {"kind": kind, "text": normalize(text).strip(), "source_locator": locator, **extras}


def script_symbol(text: str, kind: str) -> str:
    source = "0123456789+-=()"
    target = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾" if kind == "sup" else "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    if text and all(c in source for c in text):
        return text.translate(str.maketrans(source, target))
    return ("^{" if kind == "sup" else "_{") + text + "}"


def markup_text(element: Tag) -> str:
    # Preserve exponent/subscript semantics rather than concatenating 10<sup>3</sup> as 103.
    for el in element.find_all(["sup", "sub"]):
        el.replace_with(script_symbol(el.get_text(), el.name))
    for br in element.find_all("br"):
        br.replace_with("\n")
    for paragraph in element.find_all(["p", "li"]):
        paragraph.insert_before("\n")
        paragraph.insert_after("\n")
    return normalize(element.get_text())


def xml_text(element: ET.Element) -> str:
    pieces = [element.text or ""]
    for sub in element:
        value = xml_text(sub)
        if tag_name(sub) in {"LI", "LIP", "list-item"}:
            pieces.append("\n")
        pieces.append(
            script_symbol(value, tag_name(sub)) if tag_name(sub) in {"sup", "sub"} else value
        )
        pieces.append(sub.tail or "")
    return normalize("".join(pieces))


def html_blocks(data: bytes, artifact: dict[str, Any]) -> list[dict[str, Any]]:
    selector = artifact.get("selector")
    if not isinstance(selector, str) or not selector:
        raise QuarantineError("html_selector_required")
    soup = BeautifulSoup(data.decode("utf-8-sig", errors="strict"), "html.parser")
    roots = soup.select(selector)
    if len(roots) != 1:
        raise QuarantineError("html_selector_must_match_exactly_one_root")
    root = roots[0]
    source_paths: dict[int, list[int]] = {}
    source_paths[id(root)] = []
    for element in root.find_all(True):
        indices = []
        current = element
        while current is not root:
            parent = current.parent
            if not isinstance(parent, Tag):
                raise QuarantineError("html_original_parent_missing")
            siblings = parent.find_all(recursive=False)
            indices.append(next(i for i, sibling in enumerate(siblings) if sibling is current))
            current = parent
        source_paths[id(element)] = list(reversed(indices))
    for el in list(
        root.select(
            "script,style,nav,footer,form,[role=navigation],[role=banner],"
            "[aria-hidden=true],[hidden],.cookie-banner,.cookie-consent,#cookie-banner"
        )
    ):
        el.decompose()
    out = []
    block_tags = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "table", "blockquote"]
    elements = ([root] if root.name in block_tags else []) + root.find_all(block_tags)
    for i, el in enumerate(elements):
        if el.find_parent("table") or el.find_parent(["p", "li", "pre", "blockquote"]):
            continue
        locator = {
            "selector": selector,
            "element_index": i,
            "tag": el.name,
            "element_child_indices": source_paths[id(el)],
            "indices_basis": "original selected DOM before boilerplate removal",
        }
        if el.name == "table":
            rows = [
                [markup_text(cell).strip() for cell in row.find_all(["th", "td"], recursive=False)]
                for row in el.find_all("tr")
            ]
            caption = el.find("caption")
            text = (markup_text(caption).strip() + "\n") if caption else ""
            text += "\n".join("\t".join(row) for row in rows)
            out.append(block("table", text, locator, rows=rows))
        else:
            out.append(
                block(
                    "heading" if el.name.startswith("h") else "paragraph", markup_text(el), locator
                )
            )
    return out


def tag_name(el: ET.Element) -> str:
    return el.tag.rsplit("}", 1)[-1]


def xml_blocks(data: bytes) -> list[dict[str, Any]]:
    # DTD references are common in JATS; entities are never resolved/fetched.
    if b"<!ENTITY" in data.upper():
        raise QuarantineError("xml_entity_declaration_rejected")
    root = ET.fromstring(data)
    root_tag = tag_name(root)
    if root_tag not in {"article", "ARTICLE", "DOC", "RULE", "DOCUMENT"}:
        raise QuarantineError("xml_not_jats_or_federal_register")
    out: list[dict[str, Any]] = []
    allowed_heads = {"article-title", "title", "TITLE", "HD"}
    allowed_para = {"p", "P", "FP", "LI", "LIP", "list-item"}
    excluded = {"ref-list", "back", "fig", "permissions", "contrib-group", "address"}

    def visit(el: ET.Element, path: str, indices: list[int], active: bool) -> None:
        tag = tag_name(el)
        if tag in excluded:
            return
        active = active or tag in {"article-title", "abstract", "body"}
        locator = {
            "xml_path": path,
            "element_child_indices": indices,
            "path_names": "namespace-local; child indices are authoritative",
        }
        if active and tag in {"table", "GPOTABLE"}:
            table_rows = [row for row in el.iter() if tag_name(row) in {"tr", "ROW", "BOXHD"}]
            rows = [
                [
                    (
                        " ".join(xml_text(cell).split())
                        if tag_name(cell) == "CHED"
                        else xml_text(cell).strip()
                    )
                    for cell in row
                    if tag_name(cell) in {"th", "td", "ENT", "CHED"}
                ]
                for row in table_rows
            ]
            captions = [
                xml_text(child).strip() for child in el if tag_name(child) in {"TTITLE", "caption"}
            ]
            caption = "\n".join(captions)
            header_rows = sum(
                tag_name(row) == "BOXHD" or any(tag_name(c) == "th" for c in row)
                for row in table_rows
            )
            text = (caption + "\n") if caption else ""
            text += "\n".join("\t".join(row) for row in rows if row)
            out.append(
                block(
                    "table", text, locator, rows=rows, caption=caption, header_row_count=header_rows
                )
            )
            return
        if active and tag in allowed_heads | allowed_para:
            out.append(
                block("heading" if tag in allowed_heads else "paragraph", xml_text(el), locator)
            )
            return
        counters: Counter[str] = Counter()
        for index, sub in enumerate(el):
            counters[tag_name(sub)] += 1
            visit(
                sub, f"{path}/{tag_name(sub)}[{counters[tag_name(sub)]}]", [*indices, index], active
            )

    visit(root, f"/{root_tag}[1]", [], root_tag != "article")
    return out


def allowed_fields(artifact: dict[str, Any]) -> list[str]:
    paths = artifact.get("field_paths", [])
    if not paths or not all(isinstance(path, str) and path for path in paths):
        raise QuarantineError("structured_field_allowlist_required")
    for path in paths:
        names = path.split(".") if artifact["processor"] == "json" else [path]
        if any(re.sub(r"[^a-z]", "", name.lower()) in PRIVACY_FIELDS for name in names):
            raise QuarantineError("privacy_field_in_allowlist")
    return cast(list[str], paths)


def lookup(record: Any, path: str) -> Any:
    current = record
    for name in path.split("."):
        if not isinstance(current, dict) or name not in current:
            raise QuarantineError(f"missing_allowlisted_field:{path}")
        current = current[name]
    if isinstance(current, (list, dict)):
        raise QuarantineError("allowlisted_values_must_be_scalar")
    return current


def retain_record(text: str, artifact: dict[str, Any]) -> bool:
    includes, excludes = artifact.get("include_patterns", []), artifact.get("exclude_patterns", [])
    return (not includes or any(re.search(p, text, re.IGNORECASE) for p in includes)) and not any(
        re.search(p, text, re.IGNORECASE) for p in excludes
    )


def structured_blocks(
    data: bytes, artifact: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fields = allowed_fields(artifact)
    if artifact["processor"] == "csv":
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        if not reader.fieldnames or not set(fields).issubset(reader.fieldnames):
            raise QuarantineError("csv_allowlisted_headers_missing")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise QuarantineError("csv_duplicate_headers")
        records: Any = []
        line_ranges = []
        previous_line = reader.line_num
        for record in reader:
            line_ranges.append((previous_line + 1, reader.line_num))
            previous_line = reader.line_num
            if None in record or any(value is None for value in record.values()):
                raise QuarantineError("csv_malformed_row")
            records.append(record)
    else:
        records = json.loads(data.decode("utf-8-sig"), parse_float=str, parse_int=str)
        selector = artifact.get("selector")
        if selector:
            for name in selector.split("."):
                if not isinstance(records, dict) or name not in records:
                    raise QuarantineError("json_record_selector_missing")
                records = records[name]
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list):
            raise QuarantineError("json_records_must_be_array_or_object")
    out = []
    quarantined_records = []
    filtered_records = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise QuarantineError("record_must_be_object")
        values = {}
        for field in fields:
            value = record.get(field) if artifact["processor"] == "csv" else lookup(record, field)
            if value is None:
                value = "[null]"
            if isinstance(value, (dict, list)):
                raise QuarantineError("record_value_is_not_scalar")
            values[field] = normalize(json.dumps(value) if isinstance(value, bool) else str(value))
        text = "\n".join(f"{key}: {value}" for key, value in values.items())
        if not retain_record(text, artifact):
            filtered_records += 1
            continue
        locator: dict[str, Any] = {"record_index": index}
        if artifact["processor"] == "csv":
            locator.update(
                {
                    "row_number": index + 2,
                    "physical_line_start": line_ranges[index][0],
                    "physical_line_end": line_ranges[index][1],
                }
            )
        else:
            locator["selector"] = artifact.get("selector")
        if re.search(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b|\b\d{3}-\d{2}-\d{4}\b", text):
            quarantined_records.append(
                {
                    "source_locator": locator,
                    "reason": "possible_private_identifier_in_selected_record",
                }
            )
            continue
        out.append(block("record", text, locator, fields=values))
    assert len(records) == len(out) + len(quarantined_records) + filtered_records
    return out, {
        "input_records": len(records),
        "selected_records": len(out) + len(quarantined_records),
        "retained_records": len(out),
        "dropped_unselected_records": filtered_records,
        "quarantined_record_count": len(quarantined_records),
        "quarantined_records": quarantined_records,
        "field_allowlist": fields,
    }


def extract(data: bytes, artifact: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    processor = artifact["processor"]
    if processor == "html":
        return html_blocks(data, artifact), {"selector": artifact.get("selector")}
    if processor == "xml":
        return xml_blocks(data), {"xml_policy": "jats-fr-whitelist"}
    if processor in {"json", "csv"}:
        return structured_blocks(data, artifact)
    if processor == "pdf":
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise QuarantineError("encrypted_pdf")
        pages = [normalize(page.extract_text() or "") for page in reader.pages]
        if not pages:
            raise QuarantineError("pdf_has_no_pages")
        char_counts = [sum(c.isalnum() for c in text) for text in pages]
        fraction = sum(n >= 40 for n in char_counts) / len(pages)
        quality = {
            "pages": len(pages),
            "page_alphanumeric_chars": char_counts,
            "textual_page_fraction": fraction,
            "ocr_performed": False,
        }
        out = [
            block("page", f"[PDF_PAGE {i + 1}]\n{text}", {"page": i + 1})
            for i, text in enumerate(pages)
            if text.strip()
        ]
        if fraction < 0.8 or sum(char_counts) < len(pages) * 40:
            quality["quarantine_reason"] = "pdf_low_text_extraction_requires_review_or_ocr"
        return out, quality
    text = normalize(data.decode("utf-8-sig", errors="strict"))
    out = []
    for match in re.finditer(r"\S(?:[^\n]|\n(?![ \t]*\n))*", text):
        content = match.group()
        out.append(
            block(
                "heading" if content.startswith("#") else "paragraph",
                content,
                {"normalized_text_start": match.start(), "normalized_text_end": match.end()},
            )
        )
    return out, {}


def assemble(
    blocks: list[dict[str, Any]], receipt: dict[str, Any], docid: str
) -> tuple[str, list[dict[str, Any]]]:
    text = ""
    output = []
    for item in blocks:
        if not item["text"]:
            continue
        if text:
            text += "\n\n"
        start = len(text)
        text += item["text"]
        output.append(
            {
                **item,
                "document_id": docid,
                "artifact_name": receipt["name"],
                "raw_sha256": receipt["sha256"],
                "canonical_start": start,
                "canonical_end": len(text),
                "text_sha256": digest(item["text"].encode("utf-8")),
            }
        )
    if text:
        text += "\n"
    canonical_hash = digest(text.encode("utf-8"))
    for item in output:
        item["canonical_sha256"] = canonical_hash
    return text, output


def duplicate_signals(
    documents: list[dict[str, Any]], texts: dict[str, str]
) -> list[dict[str, Any]]:
    """Only signals: retain all source documents and preserve assigned families."""
    signals = []
    for i, left in enumerate(documents):
        for right in documents[i + 1 :]:
            a, b = texts[left["document_id"]], texts[right["document_id"]]
            wa, wb = a.split()[:20000], b.split()[:20000]
            sa = {tuple(wa[j : j + 5]) for j in range(max(0, len(wa) - 4))}
            sb = {tuple(wb[j : j + 5]) for j in range(max(0, len(wb) - 4))}
            similarity = len(sa & sb) / len(sa | sb) if sa and sb else 0.0
            exact = a == b
            if exact or similarity >= 0.9:
                ids = [left["document_id"], right["document_id"]]
                family = "duplicate-candidate-" + digest("|".join(sorted(ids)).encode())[:16]
                signals.append(
                    {
                        "document_ids": ids,
                        "kind": "exact" if exact else "near",
                        "five_word_shingle_jaccard": similarity,
                        "token_cap_per_document": 20000,
                        "suggested_family_id": family,
                        "review_status": "machine_candidate",
                        "sources_deleted": False,
                    }
                )
    return signals


def build(
    manifest_path: Path,
    runtime_root: Path,
    *,
    offline: bool = False,
    freeze_output: Path | None = None,
) -> dict[str, Any]:
    builder_source_sha = digest(Path(__file__).read_bytes())
    root = runtime_root.resolve()
    frozen = freeze_output.resolve() if freeze_output else root / "snapshot.json"
    if frozen.parent != root:
        raise ValueError("freeze-output must be directly inside runtime-root")
    if frozen.exists() or (root / "snapshot.json").exists():
        raise FileExistsError("snapshot already frozen: use a new runtime root; never overwrite")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    validate_manifest(manifest)
    root.mkdir(parents=True, exist_ok=True)
    frozen_manifest = child(root, "manifest.frozen.json")
    if frozen_manifest.exists() and frozen_manifest.read_bytes() != manifest_bytes:
        raise ValueError("runtime manifest changed; use a new runtime root")
    if not frozen_manifest.exists():
        with frozen_manifest.open("xb") as stream:
            stream.write(manifest_bytes)
    results, documents, unavailable = [], [], []
    texts: dict[str, str] = {}
    for doc in manifest["documents"]:
        rights = doc["rights"]
        doc_result = {
            "document_id": doc["document_id"],
            "rights": rights,
            "document_description": source_description(doc),
            "scope_status": "candidate",
            "artifacts": [],
            "status": "quarantine",
        }
        valid_content = []
        receipts = []
        for artifact in doc["artifacts"]:
            effective_rights = artifact.get("rights", rights)
            licensed = effective_rights.get("status") in {"open_license", "public_domain"}
            licensed = licensed and bool(effective_rights.get("license_url"))
            if effective_rights.get("status") == "open_license":
                licensed = licensed and bool(effective_rights.get("license_name"))
            row = {
                "name": artifact["name"],
                "role": artifact["role"],
                "processor": artifact["processor"],
                "rights": effective_rights,
                "source_description": source_description(artifact, artifact=True),
                "field_sources": {
                    "rights": "artifact" if "rights" in artifact else "document",
                    "language": "artifact" if "language" in artifact else "document",
                },
                "language": artifact.get("language", doc.get("language", "unknown")),
                "status": "quarantine",
            }
            try:
                data, receipt = raw_input(root, artifact, offline=offline)
                receipt = {
                    **receipt,
                    "rights": effective_rights,
                    "language": row["language"],
                    "role": artifact["role"],
                    "rights_eligible": licensed,
                    "source_description": row["source_description"],
                    "field_sources": row["field_sources"],
                }
                row["receipt"] = receipt
                receipts.append(receipt)
                blocks, quality = extract(data, artifact)
                text, blocks = assemble(blocks, receipt, doc["document_id"])
                row["quality"] = quality
                if not text.strip():
                    raise QuarantineError(
                        quality.get("quarantine_reason", "empty_canonical_content")
                    )
                folder = child(root, "processed", doc["document_id"])
                folder.mkdir(parents=True, exist_ok=True)
                output = child(folder, artifact["name"] + ".txt")
                output.write_bytes(text.encode("utf-8"))
                block_path = child(folder, artifact["name"] + ".blocks.jsonl")
                block_path.write_bytes(
                    b"".join(
                        (json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8")
                        for item in blocks
                    )
                )
                repeated = Counter(
                    (b["text_sha256"], json.dumps(b["source_locator"], sort_keys=True))
                    for b in blocks
                )
                row.update(
                    {
                        "processed_path": output.relative_to(root).as_posix(),
                        "blocks_path": block_path.relative_to(root).as_posix(),
                        "sha256": digest(text.encode("utf-8")),
                        "canonical_sha256": digest(text.encode("utf-8")),
                        "characters": len(text),
                        "blocks": len(blocks),
                        "quality": quality,
                        "dedup": {
                            "policy": "retain_distinct_source_locators",
                            "same_source_locator_duplicates": sum(n - 1 for n in repeated.values()),
                            "repeated_text_blocks_retained": sum(
                                n - 1 for n in Counter(b["text_sha256"] for b in blocks).values()
                            ),
                        },
                    }
                )
                reason = quality.get("quarantine_reason")
                if not licensed:
                    reason = "rights_unknown_or_license_evidence_missing"
                row["status"] = "quarantine" if reason else "processed"
                if reason:
                    row["reason"] = reason
                if not reason and artifact["role"] == "content":
                    valid_content.append((row, text))
            except Exception as error:
                row["reason"] = f"{type(error).__name__}:{error}"
                row["status"] = "missing" if isinstance(error, FileNotFoundError) else "quarantine"
            if "receipt" in row:
                row["receipt"]["processing_status"] = row["status"]
            doc_result["artifacts"].append(row)
        if valid_content:
            selected, text = valid_content[0]
            texts[doc["document_id"]] = text
            metadata = {
                key: doc.get(key)
                for key in [
                    "title",
                    "publisher",
                    "jurisdiction",
                    "language",
                    "source_type",
                    "landing_url",
                    "rights",
                    "engine_models",
                    "flight_phases",
                    "document_family_id",
                ]
            }
            metadata["rights"] = selected["rights"]
            metadata["language"] = selected["language"]
            metadata["document_declared_rights"] = rights
            metadata["document_description"] = source_description(doc)
            metadata["canonical_artifact_description"] = selected["source_description"]
            # No completeness/revision/date inheritance: PDF-wide fields cannot describe a summary.
            properties = {
                key: selected["source_description"].get(key)
                for key in [
                    "completeness",
                    "revision_status",
                    "status",
                    "source_date",
                    "publication_date",
                    "translation_status",
                    "role",
                    "artifact_key",
                    "verification",
                ]
            }
            properties.update(
                {
                    "name": selected["name"],
                    "rights": selected["rights"],
                    "language": selected["language"],
                    "processing_status": selected["status"],
                }
            )
            metadata["canonical_artifact_properties"] = properties
            metadata["field_sources"] = {
                **selected["field_sources"],
                **{
                    key: "artifact" if key in selected["source_description"] else "not_declared"
                    for key in [
                        "completeness",
                        "revision_status",
                        "status",
                        "source_date",
                        "publication_date",
                        "translation_status",
                    ]
                },
            }
            metadata["completeness"] = properties["completeness"]
            metadata["revision_status"] = properties["revision_status"]
            document = {
                "document_id": doc["document_id"],
                "processed_path": selected["processed_path"],
                "sha256": selected["sha256"],
                "canonical_sha256": selected["canonical_sha256"],
                "cohort": "refreshed",
                "freeze_cohort": "new_snapshot",
                "artifacts": receipts,
                "metadata": metadata,
                "scope_status": "candidate",
                "review_status": "machine_candidate",
                "family_reviewed": False,
                "blocks_path": selected["blocks_path"],
                "canonical_artifact_name": selected["name"],
                "alternate_artifacts": [a["name"] for a, _ in valid_content[1:]],
            }
            documents.append(document)
            doc_result["status"] = "eligible_candidate"
            doc_result["canonical_artifact_name"] = selected["name"]
        else:
            unavailable.append(doc["document_id"])
        results.append(doc_result)
    signals = duplicate_signals(documents, texts)
    summary = {
        "dataset_id": manifest["dataset_id"],
        "builder_version": BUILDER_VERSION,
        "builder_source_sha256": builder_source_sha,
        "source_manifest_sha256": digest(manifest_bytes),
        "documents": results,
        "eligible_candidate_documents": len(documents),
        "unavailable_document_ids": unavailable,
        "coverage": {
            key: dict(Counter(str(doc["metadata"].get(key, "unknown")) for doc in documents))
            for key in ["publisher", "source_type", "language", "jurisdiction"]
        },
        "format_coverage": dict(
            Counter(
                row["processor"]
                for doc in results
                for row in doc["artifacts"]
                if row["status"] == "processed"
            )
        ),
        "artifact_language_coverage": dict(
            Counter(
                row["language"]
                for doc in results
                for row in doc["artifacts"]
                if row["status"] == "processed"
            )
        ),
        "task_coverage": dict(
            Counter(
                task
                for doc in manifest["documents"]
                if doc["document_id"] in texts
                for task in doc.get("task_coverage", ["not_declared"])
            )
        ),
        "quarantine": [
            {"document_id": doc["document_id"], **row}
            for doc in results
            for row in doc["artifacts"]
            if row["status"] in {"missing", "quarantine"}
        ],
        "duplicate_family_candidates": signals,
        "notes": [
            "All new scope and labels remain machine candidates, never gold.",
            (
                "Canonical text preserves NFC/LF, numbers, units and negation; "
                "no NFKC, OCR or dehyphenation."
            ),
            (
                "First successful content artifact supplies benchmark canonical text; "
                "other formats remain separate."
            ),
            (
                "New artifacts use benchmark-compatible refreshed cohort with "
                "freeze_cohort=new_snapshot; this is no historical reproduction."
            ),
            (
                "PDF/markup source locators refer to extracted pages/elements, "
                "not original byte offsets."
            ),
            (
                "Field allowlists limit structured privacy exposure; "
                "free text still requires human privacy review."
            ),
        ],
    }
    if builder_source_sha != digest(Path(__file__).read_bytes()):
        raise RuntimeError("builder source changed during extraction; snapshot not published")
    write_json(child(root, "summary.json"), summary)
    snapshot = {
        "dataset_id": manifest["dataset_id"],
        "version": manifest["version"],
        "source_manifest_sha256": digest(manifest_bytes),
        "builder_version": BUILDER_VERSION,
        "builder_source_sha256": builder_source_sha,
        "frozen_manifest_path": "manifest.frozen.json",
        "documents": documents,
        "unavailable_document_ids": unavailable,
        "duplicate_family_candidates": signals,
        "publication_ready": False,
        "review_status": "machine_candidate",
    }
    # Exclusive create prevents an accidental frozen snapshot refresh.
    with frozen.open("xb") as stream:
        stream.write((json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--freeze-output", type=Path)
    args = parser.parse_args()
    snapshot = build(
        args.manifest, args.runtime_root, offline=args.offline, freeze_output=args.freeze_output
    )
    print(
        json.dumps(
            {
                "dataset_id": snapshot["dataset_id"],
                "eligible_candidate_documents": len(snapshot["documents"]),
                "unavailable": len(snapshot["unavailable_document_ids"]),
                "publication_ready": False,
            }
        )
    )


if __name__ == "__main__":
    main()
