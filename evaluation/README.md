# Evaluation assets

`datasets/smoke-v1.json` is a small synthetic contract dataset. It validates report matching,
claim-term recall, source-kind recall, evidence binding, and refusal behavior. It is not a paper
benchmark and must not be used to claim diagnostic accuracy.

Generate reports with a user-configured model through the application, save the returned report
objects as a JSON array, then run:

```powershell
uv run aerodiagnosis-evaluate evaluation/datasets/smoke-v1.json reports.json
```

The output records the canonical dataset hash and every per-case score, so later baselines and
ablations can be compared without silently changing the questions.
