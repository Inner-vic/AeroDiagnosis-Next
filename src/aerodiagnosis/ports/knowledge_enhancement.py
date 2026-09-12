"""Persistence boundary for model plugins and root-cause sessions."""

from __future__ import annotations

from typing import Protocol

from aerodiagnosis.domain import ModelPluginManifest, ModelPluginRecord, RootCauseSession


class KnowledgeEnhancementStore(Protocol):
    def register_model(self, manifest: ModelPluginManifest) -> ModelPluginRecord: ...

    def list_models(self) -> tuple[ModelPluginRecord, ...]: ...

    def get_model(self, model_id: str) -> ModelPluginRecord | None: ...

    def save_session(self, session: RootCauseSession) -> None: ...

    def get_session(self, rca_id: str) -> RootCauseSession | None: ...

    def list_sessions(self, *, limit: int = 50) -> tuple[RootCauseSession, ...]: ...
