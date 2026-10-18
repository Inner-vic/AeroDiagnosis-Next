"""Bounded source-language checks and proposition selection; not an entailment proof."""

from __future__ import annotations

# ruff: noqa: RUF001, SIM102 -- Chinese source grammar and exact punctuation.
import re
import unicodedata

POLICY = "source-semantics@2"
ALTERNATIVE = re.compile(
    r"要么|或者|还是|两者择一|(?:或(?!许))|之一|备选|一种可能|另一种可能|\beither\b|\bor\b", re.I
)


def translation_issues(source: str, chinese: str, context: str = "") -> list[str]:
    problems = []
    if (
        re.search(r"\bcontained (?:failure|engine failure)\b", source, re.I)
        and not re.search(r"\buncontained\b", source, re.I)
        and re.search(r"非包容|非包涵", chinese)
    ):
        problems.append("contained_failure_polarity_reversed")
    if re.search(r"\bcavitat(?:e[sd]?|ing|ion)\b", source, re.I) and not re.search(
        r"[气汽]蚀|空化", chinese
    ):
        problems.append("cavitation_terminology_missing")
    if re.search(r"\bdomestic object damage\b|foreign/domestic object damage", source, re.I):
        if re.search(r"国内|国产|本国", chinese) or not re.search(r"内部|内源", chinese):
            problems.append("internal_object_damage_mistranslated")
    if re.search(
        r"(?:a range of|several|multiple) possibilities", source + context, re.I
    ) and not ALTERNATIVE.search(chinese.splitlines()[0] if context else chinese):
        problems.append("source_alternative_scenarios_merged")
    if re.search(r"\beither\b[\s\S]{0,500}\bor\b", source, re.I) and not ALTERNATIVE.search(
        chinese
    ):
        problems.append("source_alternatives_removed")
    if re.search(
        r"(?:did not|could not|no)\s+(?:identify|find|observe)[^.!?]{0,180}"
        r"(?:that|which) would (?:indicate|suggest)",
        source,
        re.I,
    ):
        if re.search(
            r"(?:未|没有)[^。]{0,100}[，,]\s*(?:这)?(?:表明|说明|提示)[^。]{0,80}"
            r"(?:存在|有)[^。]{0,60}(?:系统性|系统级)",
            chinese,
        ):
            problems.append("negative_relative_clause_scope_lost")
    if re.search(
        r"None of[^.!?]{0,200}(?:showed|exhibited)[^.!?]{0,180}that would (?:suggest|indicate)",
        source,
        re.I,
    ) and re.search(
        r"(?:均未|未|没有)[^。]{0,120}[，,]\s*(?:这)?(?:表明|说明|提示)(?!未|无|没有|不)", chinese
    ):
        problems.append("negative_evidence_scope_lost")
    return problems


def _tokens(text: str) -> set[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    result = set(re.findall(r"[a-z0-9]{2,}", text))
    for seq in re.findall(r"[\u3400-\u9fff]+", text):
        result.update(seq[i : i + 2] for i in range(len(seq) - 1))
    return result


def relevant_propositions(statement: str, quotes: list[str]) -> list[str]:
    """Select lexical source propositions, carrying dependent qualification context.

    Every full source quote remains cited. A different proposition's modal is
    not automatically projected onto the claim. Ambiguous low-overlap matches
    retain the complete source and therefore fail conservatively.
    """
    query = _tokens(statement)
    candidates = []
    for quote in quotes:
        clauses = [x.strip() for x in re.split(r"[，,；;]|(?:\s+but\s+)", quote) if x.strip()]
        for index, clause in enumerate(clauses):
            score = len(query & _tokens(clause)) / max(1, len(query))
            context = clause
            # A pronoun, coordination, conditional or consequence may depend
            # on the uncertainty/condition of an earlier source proposition.
            if index and re.match(
                r"^(?:该|这|其|并|从而|因此|导致|造成|使|如果|若|且|和|"
                r"and |which |it |this |therefore )",
                clause,
                re.I,
            ):
                context = "，".join(clauses[: index + 1])
            # Preserve an explicit leading condition, even for a new subject.
            if index and re.match(
                r"^(?:如果|若|当|仅当|在.*条件|if |when |only if )", clauses[0], re.I
            ):
                context = clauses[0] + "，" + context
            # An either/or list scopes its members as alternatives.
            if ALTERNATIVE.search(quote) and re.search(
                r"表明|表示|可能性|indicat|possibil|either", quote, re.I
            ):
                context = quote
            candidates.append((score, context))
    if not candidates or max(x[0] for x in candidates) < 0.35:
        return quotes
    maximum = max(x[0] for x in candidates)
    return list(dict.fromkeys(text for score, text in candidates if score >= maximum - 0.04))
