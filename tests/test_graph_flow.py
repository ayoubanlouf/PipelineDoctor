"""
tests/test_graph_flow.py
========================
End-to-end tests for the PipelineDoctor LangGraph state machine,
mocking LLM and Docker sandbox calls to verify bounded loop transitions:
- Triage -> Patch -> Verify -> Done (immediate pass)
- Triage -> Patch -> Verify -> Patch -> Verify -> Done (pass on retry)
- Triage -> Patch -> Verify -> ... -> Escalate (max iterations exhausted)
- LLM parser fallback and reasoning tag stripping
- FastAPI webhook ingress verification
"""

import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from pipelinedoctor.agents.graph import create_healing_graph
from pipelinedoctor.agents.llm_client import LLMClient, extract_json, strip_think_tags
from pipelinedoctor.api.main import app
from pipelinedoctor.config import Settings
from pipelinedoctor.schemas import (
    AgentPhase,
    AgentState,
    FailingJob,
    FailureType,
    PatchProposal,
    TriageReport,
    VerificationResult,
    VerificationStatus,
    WorkflowFailureEvent,
)


@pytest.fixture
def mock_event() -> WorkflowFailureEvent:
    return WorkflowFailureEvent(
        event_id=uuid.uuid4(),
        repo="acme-corp/web-frontend",
        branch="main",
        commit_sha="a" * 40,
        run_id=987654,
        workflow_name="E2E Playwright Tests",
        failing_job=FailingJob(
            job_id=12345,
            job_name="Playwright E2E Suite",
            step_name="Run Playwright",
            conclusion="failure",
            html_url="https://github.com/acme-corp/web-frontend/actions/runs/987654",
            annotations=["Error: locator.click: locator('#submit-btn') resolved to 2 elements"],
        ),
    )


@pytest.fixture
def mock_triage_report() -> TriageReport:
    return TriageReport(
        failure_type=FailureType.SELECTOR_DRIFT,
        root_cause="The selector '#submit-btn' resolves to multiple elements after DOM update.",
        target_files=["tests/e2e/checkout.spec.ts"],
        confidence=0.95,
        evidence_snippets=["Error: locator.click: locator('#submit-btn') resolved to 2 elements"],
    )


@pytest.fixture
def mock_patch_proposal() -> PatchProposal:
    return PatchProposal(
        file_path="tests/e2e/checkout.spec.ts",
        unified_diff=(
            "--- a/tests/e2e/checkout.spec.ts\n"
            "+++ b/tests/e2e/checkout.spec.ts\n"
            "@@ -10,3 +10,3 @@\n"
            "-await page.locator('#submit-btn').click();\n"
            "+await page.locator('[data-testid=\"checkout-submit\"]').click();\n"
        ),
        explanation="Replaced ambiguous ID selector with unambiguous data-testid attribute.",
        patch_strategy="selector_update",
    )


# ---------------------------------------------------------------------------
# Test 1: Immediate Pass (Iteration 1 -> Done)
# ---------------------------------------------------------------------------
def test_graph_flow_pass_iteration_1(
    mock_event: WorkflowFailureEvent,
    mock_triage_report: TriageReport,
    mock_patch_proposal: PatchProposal,
) -> None:
    # Mock LLM
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.triage_failure.return_value = (mock_triage_report, {"total_tokens": 150})
    mock_llm.propose_patch.return_value = (mock_patch_proposal, {"total_tokens": 200})

    # Mock Sandbox: returns passed on first try
    mock_sandbox = MagicMock()
    mock_sandbox.execute_in_sandbox.return_value = VerificationResult(
        status=VerificationStatus.PASSED,
        exit_code=0,
        stdout="1 passed, 0 failed in 4.2s",
        stderr="",
        duration_seconds=4.2,
        container_id="test-box-01",
    )

    # Mock GitHub Integration
    mock_gh = MagicMock()
    mock_gh.fetch_workflow_logs.return_value = "FAIL tests/e2e/checkout.spec.ts"
    mock_gh.fetch_file_content.return_value = "await page.locator('#submit-btn').click();"
    mock_gh.create_fix_branch_and_commit.return_value = "pipelinedoctor/fix-run-987654"
    mock_gh.open_draft_pr.return_value = "https://github.com/acme-corp/web-frontend/pull/42"

    graph = create_healing_graph(
        llm_client=mock_llm,
        sandbox_executor=mock_sandbox,
        github_integration=mock_gh,
        settings=Settings(MAX_ITERATIONS=3),
    )

    initial_state: AgentState = {
        "event": mock_event,
        "current_phase": AgentPhase.TRIAGE,
        "iteration": 0,
        "max_iterations": 3,
        "patch_proposals": [],
        "applied_diffs": [],
        "verification_results": [],
        "errors": [],
    }

    final_state = graph.invoke(initial_state)

    assert final_state["current_phase"] == AgentPhase.DONE
    assert final_state["iteration"] == 1
    assert len(final_state["patch_proposals"]) == 1
    assert len(final_state["verification_results"]) == 1
    assert final_state["verification_results"][0].passed is True
    assert final_state["fix_branch"] == "pipelinedoctor/fix-run-987654"
    assert final_state["pr_url"] == "https://github.com/acme-corp/web-frontend/pull/42"
    mock_gh.open_draft_pr.assert_called_once()


# ---------------------------------------------------------------------------
# Test 2: Pass on Second Iteration (Retry Loop)
# ---------------------------------------------------------------------------
def test_graph_flow_pass_on_retry(
    mock_event: WorkflowFailureEvent,
    mock_triage_report: TriageReport,
    mock_patch_proposal: PatchProposal,
) -> None:
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.triage_failure.return_value = (mock_triage_report, {})
    mock_llm.propose_patch.return_value = (mock_patch_proposal, {})

    # Mock Sandbox: fails on iteration 1, passes on iteration 2
    mock_sandbox = MagicMock()
    mock_sandbox.execute_in_sandbox.side_effect = [
        VerificationResult(
            status=VerificationStatus.FAILED,
            exit_code=1,
            stdout="",
            stderr="SyntaxError in patch 1",
            duration_seconds=2.1,
        ),
        VerificationResult(
            status=VerificationStatus.PASSED,
            exit_code=0,
            stdout="1 passed in 3.5s",
            stderr="",
            duration_seconds=3.5,
        ),
    ]

    mock_gh = MagicMock()
    mock_gh.fetch_workflow_logs.return_value = "FAIL"
    mock_gh.fetch_file_content.return_value = "code"
    mock_gh.open_draft_pr.return_value = "https://github.com/acme-corp/web-frontend/pull/43"

    graph = create_healing_graph(
        llm_client=mock_llm,
        sandbox_executor=mock_sandbox,
        github_integration=mock_gh,
        settings=Settings(MAX_ITERATIONS=3),
    )

    initial_state: AgentState = {
        "event": mock_event,
        "current_phase": AgentPhase.TRIAGE,
        "iteration": 0,
        "max_iterations": 3,
        "patch_proposals": [],
        "applied_diffs": [],
        "verification_results": [],
    }

    final_state = graph.invoke(initial_state)

    assert final_state["current_phase"] == AgentPhase.DONE
    assert final_state["iteration"] == 2
    assert len(final_state["patch_proposals"]) == 2
    assert len(final_state["verification_results"]) == 2
    assert final_state["verification_results"][0].passed is False
    assert final_state["verification_results"][1].passed is True
    assert final_state["pr_url"] == "https://github.com/acme-corp/web-frontend/pull/43"


# ---------------------------------------------------------------------------
# Test 3: Escalation on Max Iterations Exhaustion (3 Fails)
# ---------------------------------------------------------------------------
def test_graph_flow_escalates_on_exhaustion(
    mock_event: WorkflowFailureEvent,
    mock_triage_report: TriageReport,
    mock_patch_proposal: PatchProposal,
) -> None:
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.triage_failure.return_value = (mock_triage_report, {})
    mock_llm.propose_patch.return_value = (mock_patch_proposal, {})

    # Mock Sandbox: fails every iteration
    mock_sandbox = MagicMock()
    mock_sandbox.execute_in_sandbox.return_value = VerificationResult(
        status=VerificationStatus.FAILED,
        exit_code=1,
        stdout="",
        stderr="Test still failed",
        duration_seconds=1.0,
    )

    mock_gh = MagicMock()
    mock_gh.fetch_workflow_logs.return_value = "FAIL"
    mock_gh.fetch_file_content.return_value = "code"

    graph = create_healing_graph(
        llm_client=mock_llm,
        sandbox_executor=mock_sandbox,
        github_integration=mock_gh,
        settings=Settings(MAX_ITERATIONS=3),
    )

    initial_state: AgentState = {
        "event": mock_event,
        "current_phase": AgentPhase.TRIAGE,
        "iteration": 0,
        "max_iterations": 3,
        "patch_proposals": [],
        "applied_diffs": [],
        "verification_results": [],
    }

    final_state = graph.invoke(initial_state)

    assert final_state["current_phase"] == AgentPhase.ESCALATE
    assert final_state["iteration"] == 3
    assert len(final_state["verification_results"]) == 3
    assert len(final_state["errors"]) > 0
    assert "Escalated to human reviewer" in final_state["errors"][0]
    # No PR opened on failure
    mock_gh.open_draft_pr.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: Reasoning Tag Stripper & Robust JSON Extraction
# ---------------------------------------------------------------------------
def test_strip_think_tags() -> None:
    raw_with_think = "<think>\nThinking about the problem...\nTrying approach B.\n</think>\n{\"key\": \"value\"}"
    cleaned = strip_think_tags(raw_with_think)
    assert cleaned == '{"key": "value"}'

    nested_or_multiple = "<THINK>first</THINK>content<think>second</think>"
    assert strip_think_tags(nested_or_multiple) == "content"


def test_extract_json_fenced_and_braces() -> None:
    # 1. Markdown code-fenced
    fenced = "Here is the result:\n```json\n{\"failure_type\": \"app_bug\"}\n```\nHope that helps!"
    assert extract_json(fenced) == {"failure_type": "app_bug"}

    # 2. Free text with surrounding chatter
    chatter = "Sure! Here is the JSON:\n{\"status\": \"ok\", \"code\": 200}\nLet me know if you need more."
    assert extract_json(chatter) == {"status": "ok", "code": 200}


# ---------------------------------------------------------------------------
# Test 5: FastAPI Webhook Ingress
# ---------------------------------------------------------------------------
def test_api_health() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "PipelineDoctor"}


def test_api_webhook_ignored_events() -> None:
    client = TestClient(app)

    # 1. Non-workflow_run event
    resp = client.post(
        "/webhook",
        json={"action": "opened"},
        headers={"X-GitHub-Event": "pull_request"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"

    # 2. Workflow run that succeeded (not failure)
    resp = client.post(
        "/webhook",
        json={
            "action": "completed",
            "workflow_run": {"conclusion": "success"},
        },
        headers={"X-GitHub-Event": "workflow_run"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"


def test_api_webhook_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)

    # Mock run_healing_pipeline so webhook doesn't invoke real LLM
    mock_run = MagicMock()
    monkeypatch.setattr("pipelinedoctor.api.main.run_healing_pipeline", mock_run)

    payload = {
        "action": "completed",
        "workflow_run": {
            "id": 998877,
            "name": "CI Tests",
            "conclusion": "failure",
            "head_branch": "feature/login",
            "head_sha": "b" * 40,
            "html_url": "https://github.com/org/repo/actions/runs/998877",
        },
        "repository": {
            "full_name": "org/repo",
        },
    }

    resp = client.post(
        "/webhook",
        json=payload,
        headers={"X-GitHub-Event": "workflow_run"},
    )

    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "accepted"
    assert data["run_id"] == 998877
    assert data["repo"] == "org/repo"


def test_api_heal_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    mock_run = MagicMock()
    mock_run.return_value = {
        "current_phase": AgentPhase.DONE,
        "iteration": 1,
        "fix_branch": "pipelinedoctor/fix-run-12345",
        "pr_url": "https://github.com/org/repo/pull/1",
        "errors": [],
    }
    monkeypatch.setattr("pipelinedoctor.api.main.run_healing_pipeline", mock_run)

    payload = {
        "repo": "org/repo",
        "branch": "main",
        "commit_sha": "a" * 40,
        "run_id": 12345,
    }
    resp = client.post("/heal", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["phase"] == "done"
    assert data["pr_url"] == "https://github.com/org/repo/pull/1"
    assert data["repo"] == "org/repo"

