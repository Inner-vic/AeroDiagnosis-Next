from aerodiagnosis.application.agent_models import VerificationDecision, coverage_issues


def test_verifier_cannot_approve_absent_question_coverage():
    decision = VerificationDecision(sufficient=True, supported_claim_indexes=(0,))
    assert coverage_issues(decision, 1)


def test_coverage_cannot_point_to_unknown_or_unsupported_claim():
    decision = VerificationDecision(
        sufficient=True,
        supported_claim_indexes=(0,),
        question_coverage=({"requirement": "cause", "covered": True, "claim_indexes": [1]},),
    )
    assert coverage_issues(decision, 1)


def test_coverage_accepts_verified_candidate_claim():
    decision = VerificationDecision(
        sufficient=True,
        supported_claim_indexes=(0,),
        question_coverage=({"requirement": "cause", "covered": True, "claim_indexes": [0]},),
    )
    assert not coverage_issues(decision, 1)
