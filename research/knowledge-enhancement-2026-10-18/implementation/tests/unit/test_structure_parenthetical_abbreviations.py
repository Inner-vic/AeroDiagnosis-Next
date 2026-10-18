from aerodiagnosis.ingestion import ParserRegistry


def test_wrapped_service_bulletin_abbreviation_stays_in_its_sentence():
    text = (
        "This requirement applies to the model CF34-8C5 engine listed in Service Bulletin\n"
        "(SB) 72-0123 dated June 1, 2024."
    )
    parsed = ParserRegistry(
        text_strategy="hierarchical", parent_budget=3200, child_budget=600
    ).parse("source.txt", text.encode())
    assert [c.content for c in parsed.chunks] == [text]
    assert parsed.canonical_content.decode() == text


def test_real_regulatory_list_markers_still_form_lists():
    text = (
        "(1) Remove affected parts.\n(i) Inspect.\n(ii) Replace if cracked.\n"
        "(A) Retain eligibility."
    )
    parsed = ParserRegistry(
        text_strategy="hierarchical", parent_budget=3200, child_budget=600
    ).parse("source.txt", text.encode())
    assert [c.content for c in parsed.chunks] == [text]
    assert parsed.chunks[0].locator.canonical()["coordinates"]["unit_type"] == "list"
