from __future__ import annotations

from pathlib import Path

from aerodiagnosis.adapters.api import create_app


def _static_root() -> Path:
    return Path(__file__).parents[2] / "src" / "aerodiagnosis" / "adapters" / "web" / "static"


def test_frontend_exposes_focused_knowledge_qa_and_progressive_root_cause_flow() -> None:
    html = (_static_root() / "index.html").read_text(encoding="utf-8")
    script = (_static_root() / "app.js").read_text(encoding="utf-8")

    assert 'id="apiKey"' in html
    assert 'id="sessionList"' in html
    assert "运行检查器" not in html
    assert 'id="parameterEditor"' not in html
    assert 'id="qaAttachmentInput"' in html
    assert 'id="voiceQuestion"' in html
    assert 'data-rca-stage="1"' in html
    assert 'data-rca-stage="4"' in html
    assert 'data-theme-choice="paper"' in html
    assert 'data-theme-choice="lab"' in html
    assert 'data-module="knowledge"' in html
    assert 'data-module="rootcause"' in html
    assert 'data-module="models"' in html
    assert 'id="knowledgeFile"' in html
    assert 'id="rcaFile"' in html
    assert 'id="agentLoop"' in html
    assert 'id="modelGrid"' in html
    assert 'id="graphChart"' in html
    assert 'id="resetGraph"' in html
    assert '/assets/vendor/echarts.min.js' in html
    assert 'id="caseGrid"' in html
    assert 'id="providerModeDefault"' in html
    assert "sessionStorage.setItem('aero.provider'" in script
    assert "localStorage.setItem('aero.sessions'" in script
    assert "api_key:state.provider" not in script
    assert "/diagnoses" in script
    assert "/documents" in script
    assert "/graph?" in script
    assert "/cases?" in script
    assert "/root-cause-sessions" in script
    assert "/models" in script
    assert "demoRcaCsv" in script
    assert "addQaAttachments" in script
    assert "startVoiceInput" in script
    assert "showRcaStage" in script
    assert "echarts.init" in script
    assert "layout:'force'" in script
    assert "roam:true" in script
    assert "draggable:true" in script
    assert "replace(/^DEMO-/,'')" in script
    assert "Number.isFinite(value)" in script
    assert "parameters()" not in script
    assert "body.provider=state.provider" in script


def test_application_serves_the_new_frontend_at_root() -> None:
    app = create_app()

    assert "/" in {getattr(route, "path", None) for route in app.routes}
    assert "assets" in {getattr(route, "name", None) for route in app.routes}
