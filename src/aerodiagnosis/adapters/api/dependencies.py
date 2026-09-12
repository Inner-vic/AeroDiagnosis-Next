"""Trusted execution context for HTTP requests."""

from __future__ import annotations

import secrets
from typing import Annotated, cast

from fastapi import Header, HTTPException, Request, status

from aerodiagnosis.bootstrap import Application


def get_application(request: Request) -> Application:
    return cast(Application, request.app.state.application)


def require_operator(
    request: Request,
    operator_token: Annotated[str | None, Header(alias="X-AeroDiagnosis-Operator-Token")] = None,
) -> Application:
    application = get_application(request)
    expected = application.settings.operator_token
    if expected is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="document writes are disabled until AERODIAGNOSIS_OPERATOR_TOKEN is set",
        )
    if operator_token is None or not secrets.compare_digest(operator_token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="valid operator token required",
        )
    return application
