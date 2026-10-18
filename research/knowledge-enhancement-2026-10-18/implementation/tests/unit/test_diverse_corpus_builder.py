from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from pypdf import PdfWriter

MODULE = Path(__file__).resolve().parents[2] / "scripts/build_diverse_corpus.py"
spec = importlib.util.spec_from_file_location("diverse_corpus_test", MODULE)
assert spec and spec.loader
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)


def artifact(name: str = "source.txt", processor: str = "text", **kwargs: Any) -> dict[str, Any]:
    media = {
        "text": "text/plain",
        "csv": "text/csv",
        "json": "application/json",
        "html": "text/html",
        "xml": "application/xml",
        "pdf": "application/pdf",
    }
    return {
        "name": name,
        "url": "https://example.org/" + name,
        "role": "content",
        "media_type": media[processor],
        "processor": processor,
        "selector": None,
        "field_paths": [],
        "include_patterns": [],
        "exclude_patterns": [],
        **kwargs,
    }


def document(
    docid: str, artifacts: list[dict[str, Any]], *, rights: str = "public_domain"
) -> dict[str, Any]:
    return {
        "document_id": docid,
        "title": docid,
        "publisher": "synthetic test fixture",
        "jurisdiction": "test",
        "language": "en",
        "source_type": "manual",
        "landing_url": "https://example.org/",
        "engine_models": [],
        "flight_phases": [],
        "scope_status": "candidate",
        "document_family_id": "family-" + docid,
        "rights": {
            "status": rights,
            "license_url": "https://example.org/rights",
            "license_name": "fixture-only",
        },
        "artifacts": artifacts,
    }


def setup(tmp_path: Path, docs: list[dict[str, Any]], raw: dict[str, bytes]) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"dataset_id": "fixture-v2", "version": "2.0", "documents": docs}),
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime"
    (runtime / "raw").mkdir(parents=True)
    for name, data in raw.items():
        (runtime / "raw" / name).write_bytes(data)
    return manifest, runtime


def test_nfc_lf_units_negation_exact_spans_and_raw_immutable(tmp_path: Path) -> None:
    raw = "# Engine\r\n\r\nNo stall at 12㎏ and 10² K. cafe\u0301 is not repaired.\r\n".encode()
    manifest, root = setup(tmp_path, [document("D1", [artifact()])], {"source.txt": raw})
    snapshot = builder.build(manifest, root, offline=True)
    doc = snapshot["documents"][0]
    canonical = (root / doc["processed_path"]).read_bytes()
    assert b"\r" not in canonical
    text = canonical.decode()
    assert "12㎏" in text and "10²" in text and "café" in text and "not repaired" in text
    assert (root / "raw/source.txt").read_bytes() == raw
    assert doc["cohort"] == "refreshed" and doc["freeze_cohort"] == "new_snapshot"
    assert snapshot["publication_ready"] is False
    for line in (root / doc["blocks_path"]).read_text(encoding="utf-8").splitlines():
        b = json.loads(line)
        assert text[b["canonical_start"] : b["canonical_end"]] == b["text"]
        assert b["canonical_sha256"] == doc["canonical_sha256"]
        assert b["raw_sha256"] == builder.digest(raw)
    with pytest.raises(FileExistsError):
        builder.build(manifest, root, offline=True)
    with pytest.raises(FileExistsError):
        builder.build(manifest, root, offline=True, freeze_output=root / "alternate.json")


def test_unknown_rights_and_missing_isolate_without_network(
    tmp_path: Path, monkeypatch: Any
) -> None:
    docs = [
        document("UNKNOWN", [artifact()], rights="unknown"),
        document("MISSING", [artifact("missing.txt")]),
        document("OK", [artifact("ok.txt")]),
    ]
    manifest, root = setup(
        tmp_path, docs, {"source.txt": b"Engine does not stall.", "ok.txt": b"No icing."}
    )
    monkeypatch.setattr(builder, "download", lambda _: pytest.fail("offline contacted network"))
    result = builder.build(manifest, root, offline=True)
    assert [d["document_id"] for d in result["documents"]] == ["OK"]
    assert result["unavailable_document_ids"] == ["UNKNOWN", "MISSING"]
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["quarantine"]) == 2
    assert summary["quarantine"][0]["reason"] == "rights_unknown_or_license_evidence_missing"
    assert summary["quarantine"][1]["status"] == "missing"


def test_html_selector_tables_inline_units_and_boilerplate() -> None:
    data = (
        b'<html><nav>outer nav</nav><article id="body"><h1>Engine</h1>'
        b"<nav>inner nav</nav><script>bad()</script>"
        b'<div class="cookie-banner">cookies</div>'
        b"<p>No stall at 10<sup>3</sup> K.</p><table><caption>Limits</caption>"
        b"<tr><th>Stage</th><th>Pressure</th></tr><tr><td>Climb</td>"
        b"<td>50 kPa</td></tr></table></article></html>"
    )
    rows = builder.html_blocks(data, artifact("x.html", "html", selector="#body"))
    assert [b["kind"] for b in rows] == ["heading", "paragraph", "table"]
    assert "10³ K" in rows[1]["text"]
    assert rows[2]["rows"] == [["Stage", "Pressure"], ["Climb", "50 kPa"]]
    assert not any(word in "".join(b["text"] for b in rows) for word in ["cookies", "bad()", "nav"])
    for selector in [None, ".missing", "p,table"]:
        with pytest.raises(builder.QuarantineError):
            builder.html_blocks(data, artifact("x.html", "html", selector=selector))


def test_xml_whitelist_keeps_inline_negation_table_and_excludes_references() -> None:
    data = (
        b"<article><front><article-meta><title-group>"
        b"<article-title>Engine</article-title></title-group><abstract>"
        b"<p>No ice.</p></abstract></article-meta></front><body><sec>"
        b"<title>Limits</title><p>Do <italic>not</italic> exceed 10 kPa.</p>"
        b"<table-wrap><table><tr><th>Unit</th><td>kPa</td></tr></table>"
        b"</table-wrap></sec></body><back><ref-list><p>ignore citation</p>"
        b"</ref-list></back></article>"
    )
    rows = builder.xml_blocks(data)
    text = "\n".join(b["text"] for b in rows)
    assert "Do not exceed 10 kPa." in text and "ignore citation" not in text
    assert any(b["kind"] == "table" and b["rows"] == [["Unit", "kPa"]] for b in rows)
    with pytest.raises(builder.QuarantineError):
        builder.xml_blocks(b"<html><p>Server error</p></html>")
    with pytest.raises(builder.QuarantineError):
        builder.xml_blocks(
            b'<!DOCTYPE article [<!ENTITY x "secret">]><article><body><p>&x;</p></body></article>'
        )


def test_csv_allowlist_filter_multiline_position_and_privacy() -> None:
    raw = (
        b'JASC,Discrepancy,Email\n7200,"No stall.\n12 kPa.",pilot@example.org\n'
        b"2700,Control fault,x@example.org\n7200,Excluded event,x@example.org\n"
    )
    a = artifact(
        "x.csv",
        "csv",
        field_paths=["JASC", "Discrepancy"],
        include_patterns=["7200"],
        exclude_patterns=["Excluded"],
    )
    rows, quality = builder.structured_blocks(raw, a)
    assert len(rows) == 1 and quality["input_records"] == 3
    assert "Email" not in rows[0]["text"] and "pilot@" not in rows[0]["text"]
    assert rows[0]["source_locator"]["physical_line_start"] == 2
    assert rows[0]["source_locator"]["physical_line_end"] == 3
    assert "No stall.\n12 kPa." in rows[0]["text"]
    with pytest.raises(builder.QuarantineError, match="privacy"):
        builder.structured_blocks(raw, artifact("x.csv", "csv", field_paths=["Email"]))
    with pytest.raises(builder.QuarantineError, match="allowlist"):
        builder.structured_blocks(raw, artifact("x.csv", "csv"))
    with pytest.raises(builder.QuarantineError, match="headers"):
        builder.structured_blocks(raw, artifact("x.csv", "csv", field_paths=["unknown"]))


def test_json_allowlist_nested_record_location_and_no_implicit_full_dump() -> None:
    data = json.dumps(
        {
            "results": [
                {
                    "engine": {"model": "CFM56"},
                    "note": "not stalled",
                    "Email": "private@example.org",
                },
                {"engine": {"model": "other"}, "note": "ignore"},
            ]
        }
    ).encode()
    a = artifact(
        "x.json",
        "json",
        selector="results",
        field_paths=["engine.model", "note"],
        include_patterns=["CFM56"],
    )
    rows, _ = builder.structured_blocks(data, a)
    assert rows[0]["source_locator"] == {"record_index": 0, "selector": "results"}
    assert "not stalled" in rows[0]["text"] and "Email" not in rows[0]["text"]
    with pytest.raises(builder.QuarantineError, match="missing_allowlisted"):
        builder.structured_blocks(data, {**a, "field_paths": ["engine.missing"]})
    with pytest.raises(builder.QuarantineError, match="selector"):
        builder.structured_blocks(data, {**a, "selector": "absent"})


def test_selected_free_text_private_identifier_is_quarantined() -> None:
    rows, quality = builder.structured_blocks(
        b'[{"note":"Contact pilot@example.org"},{"note":"No stall at 12 kPa."}]',
        artifact("x.json", "json", field_paths=["note"]),
    )
    assert len(rows) == 1 and rows[0]["source_locator"]["record_index"] == 1
    assert quality["quarantined_record_count"] == 1
    assert quality["quarantined_records"][0]["source_locator"]["record_index"] == 0
    assert "pilot@example.org" not in json.dumps(quality)


def test_pdf_blank_is_quarantined_with_page_quality_no_fake_ocr(tmp_path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    stream = io.BytesIO()
    writer.write(stream)
    blocks, quality = builder.extract(stream.getvalue(), artifact("blank.pdf", "pdf"))
    assert blocks == [] and quality["pages"] == 1 and quality["ocr_performed"] is False
    assert quality["quarantine_reason"].startswith("pdf_low_text")
    manifest, root = setup(
        tmp_path,
        [document("BLANK", [artifact("blank.pdf", "pdf")])],
        {"blank.pdf": stream.getvalue()},
    )
    snapshot = builder.build(manifest, root, offline=True)
    assert snapshot["documents"] == []


def test_receipt_hash_drift_never_redownloads_or_overwrites(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _manifest, root = setup(tmp_path, [document("D1", [artifact()])], {"source.txt": b"No stall."})
    builder.raw_input(root, artifact(), offline=True)
    (root / "raw/source.txt").write_bytes(b"Modified")
    monkeypatch.setattr(builder, "download", lambda _: pytest.fail("silent refresh"))
    with pytest.raises(builder.QuarantineError, match="hash_drift"):
        builder.raw_input(root, artifact(), offline=False)
    assert (root / "raw/source.txt").read_bytes() == b"Modified"
    (root / "raw/source.txt").unlink()
    with pytest.raises(builder.QuarantineError, match="no_automatic_refresh"):
        builder.raw_input(root, artifact(), offline=False)


@pytest.mark.parametrize("name", ["../secret", "C:\\secret", "x/y", "CON.txt", "x.", ".."])
def test_manifest_path_traversal_rejected_before_io(tmp_path: Path, name: str) -> None:
    manifest, root = setup(tmp_path, [document("D1", [artifact(name)])], {})
    with pytest.raises(ValueError, match="unsafe"):
        builder.build(manifest, root, offline=True)


def test_media_error_page_and_manifest_hash_mismatch() -> None:
    with pytest.raises(builder.QuarantineError, match="magic"):
        builder.verify_media(b"<html>Error</html>", artifact("x.pdf", "pdf"), "application/pdf")
    with pytest.raises(builder.QuarantineError, match="media_type"):
        builder.verify_media(b"%PDF-test", artifact("x.pdf", "pdf"), "text/html")
    with pytest.raises(builder.QuarantineError, match="error_page"):
        builder.verify_media(b"<html>Error</html>", artifact(), "text/plain")


def test_alternate_formats_retained_canonical_not_concatenated_and_duplicates_signaled(
    tmp_path: Path,
) -> None:
    docs = [
        document("ONE", [artifact("one.txt"), artifact("two.txt")]),
        document("OTHER", [artifact("other.txt")]),
    ]
    raw = b"Engine does not stall at 10 kPa."
    manifest, root = setup(
        tmp_path, docs, {name: raw for name in ["one.txt", "two.txt", "other.txt"]}
    )
    snap = builder.build(manifest, root, offline=True)
    assert len(snap["documents"]) == 2
    assert snap["documents"][0]["alternate_artifacts"] == ["two.txt"]
    assert (root / snap["documents"][0]["processed_path"]).read_text().count("Engine") == 1
    assert snap["duplicate_family_candidates"][0]["kind"] == "exact"
    assert snap["duplicate_family_candidates"][0]["sources_deleted"] is False
    assert snap["documents"][0]["metadata"]["document_family_id"] == "family-ONE"


def test_same_text_on_different_pages_retained_with_distinct_locators() -> None:
    rows = [
        builder.block("page", "No stall", {"page": 1}),
        builder.block("page", "No stall", {"page": 2}),
    ]
    text, rows = builder.assemble(rows, {"name": "x.pdf", "sha256": "a" * 64}, "D")
    assert text.count("No stall") == 2
    assert rows[0]["source_locator"] != rows[1]["source_locator"]


def test_https_and_size_bounds(tmp_path: Path, monkeypatch: Any) -> None:
    for url in ["http://example.org/x", "https://user:password@example.org/x", "file:///x"]:
        with pytest.raises(ValueError):
            builder.download(url)

    class Headers:
        def get(self, key: str) -> str | None:
            return "100000000" if key == "Content-Length" else None

        def get_content_type(self) -> str:
            return "text/plain"

    class Response:
        headers = Headers()

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def geturl(self) -> str:
            return "https://example.org/x"

    class Opener:
        def open(self, *_: object, **__: object) -> Response:
            return Response()

    monkeypatch.setattr(builder.urllib.request, "build_opener", lambda *_: Opener())
    with pytest.raises(builder.QuarantineError, match="too_large"):
        builder.download("https://example.org/x")


def test_artifact_rights_override_never_licenses_unknown_pdf_and_language_counts(
    tmp_path: Path,
) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    stream = io.BytesIO()
    writer.write(stream)
    html = artifact(
        "summary.html",
        "html",
        selector="article",
        language="en",
        rights={
            "status": "open_license",
            "license_name": "OGL 3",
            "license_url": "https://example.org/ogl",
        },
    )
    doc = document("UK", [artifact("full.pdf", "pdf"), html], rights="unknown")
    doc["language"] = "fr"
    manifest, root = setup(
        tmp_path,
        [doc],
        {
            "full.pdf": stream.getvalue(),
            "summary.html": b"<article><h1>Engine incident</h1><p>No stall.</p></article>",
        },
    )
    snap = builder.build(manifest, root, offline=True)
    assert snap["documents"][0]["canonical_artifact_name"] == "summary.html"
    assert snap["documents"][0]["metadata"]["rights"]["status"] == "open_license"
    assert snap["documents"][0]["metadata"]["document_declared_rights"]["status"] == "unknown"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["coverage"]["language"] == {"en": 1}
    assert summary["documents"][0]["artifacts"][0]["status"] == "quarantine"


def test_xml_original_child_indices_can_resolve_each_extracted_element() -> None:
    from xml.etree import ElementTree as ET

    data = (
        b"<article><front><article-meta><title-group>"
        b"<article-title>Engine</article-title></title-group><abstract>"
        b"<p>Do not exceed 10<sup>3</sup> K.</p></abstract></article-meta>"
        b"</front><body><sec><title>Limits</title><p>No ice.</p><p>Second.</p>"
        b"</sec></body></article>"
    )
    root = ET.fromstring(data)
    blocks = builder.xml_blocks(data)
    assert any("10³" in row["text"] for row in blocks)
    for row in blocks:
        el = root
        for index in row["source_locator"]["element_child_indices"]:
            el = list(el)[index]
        assert builder.xml_text(el).strip() == row["text"]
        assert "scope[" not in row["source_locator"]["xml_path"]


def test_receipt_source_relabel_and_expected_hash_rejected(tmp_path: Path) -> None:
    _, root = setup(tmp_path, [document("D", [artifact()])], {"source.txt": b"No ice."})
    builder.raw_input(root, artifact(), offline=True)
    with pytest.raises(builder.QuarantineError, match="url_changed"):
        builder.raw_input(root, artifact(url="https://other.example.org/source.txt"), offline=True)
    with pytest.raises(builder.QuarantineError, match="hash_mismatch"):
        builder.raw_input(root, artifact(sha256="0" * 64), offline=True)


def test_plain_transport_html_pre_is_explicit_and_remains_html() -> None:
    raw = b"<html><body><pre>No stall.\n\nLimit 12 kPa.</pre></body></html>"
    a = artifact("fr.html", "html", selector="body > pre", transport_content_types=["text/plain"])
    builder.verify_media(raw, a, "text/plain")
    rows = builder.html_blocks(raw, a)
    assert rows and "No stall." in rows[0]["text"] and "12 kPa" in rows[0]["text"]
    with pytest.raises(builder.QuarantineError, match="media_type"):
        builder.verify_media(raw, {**a, "transport_content_types": []}, "text/plain")


def test_csv_bad_private_row_does_not_abort_good_record_and_numbers_preserved() -> None:
    data = b"JASC,Note\n7200,Contact pilot@example.org\n7200,No ice at 12 kPa\n"
    rows, quality = builder.structured_blocks(
        data, artifact("x.csv", "csv", field_paths=["JASC", "Note"])
    )
    assert len(rows) == 1 and rows[0]["source_locator"]["row_number"] == 3
    assert quality["quarantined_record_count"] == 1 and quality["dropped_unselected_records"] == 0
    assert "pilot@example.org" not in json.dumps(quality)
    rows, _ = builder.structured_blocks(
        b'[{"number":1.2300e-3,"present":false}]',
        artifact("x.json", "json", field_paths=["number", "present"]),
    )
    assert rows[0]["fields"] == {"number": "1.2300e-3", "present": "false"}


def test_selected_summary_metadata_does_not_inherit_pdf_completeness_or_revision(
    tmp_path: Path,
) -> None:
    summary_artifact = artifact(
        "summary.html",
        "html",
        selector="article",
        completeness="summary",
        rights={
            "status": "open_license",
            "license_name": "OGL 3",
            "license_url": "https://example.org/ogl",
        },
        acquisition={"status": "downloaded", "body": "large acquisition payload"},
    )
    doc = document(
        "AAIB",
        [
            artifact("unknown.txt", completeness="full_report", revision_status="final"),
            summary_artifact,
        ],
        rights="unknown",
    )
    doc.update(
        {
            "completeness": "full_report",
            "revision_status": "amended_2023",
            "publication_date": "2021-05-06",
            "topic_tags": ["wrong_engine_manual"],
            "retrieval_policy": {"scope": "reference_only"},
            "acquisition_summary": {"large": "download payload"},
        }
    )
    manifest, root = setup(
        tmp_path,
        [doc],
        {
            "unknown.txt": b"Full report. No surge.",
            "summary.html": b"<article><p>Incident summary.</p></article>",
        },
    )
    snap = builder.build(manifest, root, offline=True)
    metadata = snap["documents"][0]["metadata"]
    assert metadata["canonical_artifact_description"]["completeness"] == "summary"
    assert metadata["canonical_artifact_properties"]["completeness"] == "summary"
    assert metadata["canonical_artifact_properties"]["revision_status"] is None
    assert metadata["canonical_artifact_properties"]["publication_date"] is None
    assert metadata["canonical_artifact_properties"]["role"] == "content"
    assert metadata["field_sources"]["rights"] == "artifact"
    assert metadata["field_sources"]["revision_status"] == "not_declared"
    assert metadata["document_description"]["revision_status"] == "amended_2023"
    assert metadata["document_description"]["retrieval_policy"] == {"scope": "reference_only"}
    assert "acquisition_summary" not in metadata["document_description"]
    assert "acquisition" not in metadata["canonical_artifact_description"]


def test_historical_directive_restrictions_survive_without_cancellation_metadata_overwrite(
    tmp_path: Path,
) -> None:
    doc = document(
        "EASA",
        [
            artifact(
                "ad.txt",
                completeness="full_directive",
                revision_status="superseded",
                publication_date="2010-11-10",
            ),
            artifact(
                "cancel.txt",
                role="metadata",
                completeness="cancellation_notice_metadata",
                publication_date="2011-03-02",
            ),
        ],
    )
    doc.update(
        {
            "revision_status": "superseded_by_2010-0242-e",
            "regulatory_use": "historical_only_not_current_maintenance_instruction",
            "retrieval_policy": {"allow_current_maintenance": False},
            "applicability_status": "candidate_hints_not_verified_applicability",
            "related_family_id": "report-2010",
            "notes": ["Do not use as current instructions."],
        }
    )
    manifest, root = setup(
        tmp_path,
        [doc],
        {"ad.txt": b"Historical inspection instructions.", "cancel.txt": b"Cancellation notice."},
    )
    snap = builder.build(manifest, root, offline=True)
    d = snap["documents"][0]
    m = d["metadata"]
    assert m["document_description"]["regulatory_use"] == doc["regulatory_use"]
    assert m["document_description"]["retrieval_policy"] == {"allow_current_maintenance": False}
    assert m["canonical_artifact_properties"]["publication_date"] == "2010-11-10"
    assert m["canonical_artifact_properties"]["role"] == "content"
    assert d["artifacts"][1]["source_description"]["publication_date"] == "2011-03-02"
    assert d["artifacts"][1]["source_description"]["role"] == "metadata"
    assert d["artifacts"][1]["source_description"]["completeness"] == "cancellation_notice_metadata"


def test_original_translation_identity_retained_per_artifact(tmp_path: Path) -> None:
    doc = document(
        "BEA",
        [
            artifact(
                "fr.txt",
                language="fr",
                completeness="full_report",
                translation_status="original_reference",
            ),
            artifact(
                "en.txt",
                language="en",
                completeness="full_report",
                translation_status="courtesy_translation_french_is_reference",
            ),
        ],
    )
    doc.update(
        {
            "language": "fr",
            "languages": ["fr", "en"],
            "source_date": "2020-09",
            "event_date": "2017-09-30",
        }
    )
    manifest, root = setup(
        tmp_path, [doc], {"fr.txt": b"Rapport de reference.", "en.txt": b"Courtesy translation."}
    )
    snap = builder.build(manifest, root, offline=True)
    d = snap["documents"][0]
    assert (
        d["metadata"]["canonical_artifact_properties"]["translation_status"] == "original_reference"
    )
    assert d["metadata"]["language"] == "fr"
    assert d["metadata"]["document_description"]["languages"] == ["fr", "en"]
    assert d["metadata"]["document_description"]["event_date"] == "2017-09-30"
    assert d["artifacts"][1]["language"] == "en"
    assert d["artifacts"][1]["source_description"]["translation_status"].startswith(
        "courtesy_translation"
    )


def test_manifest_exact_bytes_frozen_and_changed_partial_runtime_manifest_rejected(
    tmp_path: Path,
) -> None:
    manifest, root = setup(tmp_path, [document("D", [artifact()])], {"source.txt": b"No stall."})
    payload = manifest.read_bytes()
    snap = builder.build(manifest, root, offline=True)
    assert (root / snap["frozen_manifest_path"]).read_bytes() == payload
    assert builder.digest(payload) == snap["source_manifest_sha256"]
    assert snap["builder_source_sha256"] == builder.digest(MODULE.read_bytes())
    assert snap["builder_version"].startswith("diverse-corpus@3:")
    # A partial failed run still cannot be resumed against a changed manifest.
    other_root = tmp_path / "partial"
    other_root.mkdir()
    (other_root / "manifest.frozen.json").write_bytes(payload)
    changed = json.loads(payload)
    changed["version"] = "changed"
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="manifest changed"):
        builder.build(manifest, other_root, offline=True)


def test_record_partitions_are_exclusive_selected_is_retained_plus_quarantined() -> None:
    data = b"JASC,Note\n7200,Contact pilot@example.org\n7200,No ice\n2700,Flight controls\n"
    rows, q = builder.structured_blocks(
        data,
        artifact(
            "x.csv", "csv", field_paths=["JASC", "Note"], include_patterns=[r"(?m)^JASC: 7200$"]
        ),
    )
    assert len(rows) == q["retained_records"] == 1
    assert q["input_records"] == 3
    assert q["selected_records"] == 2
    assert q["quarantined_record_count"] == 1
    assert q["dropped_unselected_records"] == 1
    assert (
        q["input_records"]
        == q["retained_records"] + q["quarantined_record_count"] + q["dropped_unselected_records"]
    )
    assert q["selected_records"] == q["retained_records"] + q["quarantined_record_count"]


def test_federal_register_table_preserves_caption_column_headers_and_values() -> None:
    data = (
        b"<DOC><GPOTABLE><TTITLE>Estimated Costs</TTITLE><BOXHD>"
        b"<CHED>Action</CHED><CHED>Parts cost</CHED><CHED>Cost per<LI>product</LI></CHED>"
        b"</BOXHD><ROW><ENT>Stage 4</ENT><ENT>442,800</ENT><ENT>455,550</ENT></ROW>"
        b"<ROW><ENT>Stage 5</ENT><ENT>205,100</ENT><ENT>217,850</ENT></ROW>"
        b"</GPOTABLE></DOC>"
    )
    result = builder.xml_blocks(data)
    assert len(result) == 1
    table = result[0]
    assert table["caption"] == "Estimated Costs" and table["header_row_count"] == 1
    assert table["rows"][0] == ["Action", "Parts cost", "Cost per product"]
    assert "Parts cost" in table["text"] and "455,550" in table["text"]
    assert table["rows"][1][1:] == ["442,800", "455,550"]
    assert table["source_locator"]["element_child_indices"] == [0]
