"""
pipelinedoctor/api/main.py
==========================
FastAPI Control Plane and Webhook Receiver for PipelineDoctor.
Listens for GitHub Actions workflow_run failures, deserializes payloads
into WorkflowFailureEvent schemas, and triggers the LangGraph state machine.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from pipelinedoctor.agents.graph import create_healing_graph
from pipelinedoctor.config import get_settings
from pipelinedoctor.schemas import (
    AgentPhase,
    AgentState,
    FailingJob,
    WorkflowFailureEvent,
)

app = FastAPI(
    title="PipelineDoctor",
    description="Autonomous, event-driven Self-Healing CI/CD & QA agent system",
    version="0.1.0",
)


def verify_github_signature(payload: bytes, secret: str, signature_header: str | None) -> bool:
    """Verifies that the webhook payload was signed by GitHub using HMAC-SHA256."""
    if not secret:
        return True
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    received = signature_header.removeprefix("sha256=")
    return hmac.compare_digest(expected, received)


def run_healing_pipeline(event: WorkflowFailureEvent) -> AgentState:
    """Executes the self-healing LangGraph state machine for a failure event."""
    settings = get_settings()
    graph = create_healing_graph(settings=settings)

    initial_state: AgentState = {
        "event": event,
        "current_phase": AgentPhase.TRIAGE,
        "iteration": 0,
        "max_iterations": settings.MAX_ITERATIONS,
        "patch_proposals": [],
        "applied_diffs": [],
        "verification_results": [],
        "errors": [],
        "llm_call_count": 0,
        "fetched_file_contents": {},
    }

    final_state: AgentState = graph.invoke(initial_state)
    return final_state


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "service": "PipelineDoctor"}


@app.post("/heal")
def manual_heal(event: WorkflowFailureEvent) -> dict[str, Any]:
    """Directly triggers the self-healing pipeline with a WorkflowFailureEvent."""
    final_state = run_healing_pipeline(event)
    return {
        "status": "completed",
        "phase": str(final_state.get("current_phase")),
        "repo": event.repo,
        "run_id": event.run_id,
        "commit_sha": event.commit_sha,
        "iteration": final_state.get("iteration"),
        "fix_branch": final_state.get("fix_branch"),
        "pr_url": final_state.get("pr_url"),
        "errors": final_state.get("errors"),
    }


@app.post("/webhook")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: str | None = Header(None, alias="X-GitHub-Event"),
    x_hub_signature_256: str | None = Header(None, alias="X-Hub-Signature-256"),
) -> JSONResponse:
    """
    Receives GitHub Actions webhook payloads.
    Triggers the healing state machine on `workflow_run` failure events.
    """
    settings = get_settings()
    body_bytes = await request.body()

    # 1. Signature Verification
    if settings.GITHUB_WEBHOOK_SECRET:
        if not verify_github_signature(body_bytes, settings.GITHUB_WEBHOOK_SECRET, x_hub_signature_256):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid GitHub HMAC-SHA256 signature",
            )

    # 2. Check Event Type
    if x_github_event != "workflow_run":
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ignored", "reason": f"Event '{x_github_event}' is not workflow_run"},
        )

    payload = await request.json()
    action = payload.get("action")
    workflow_run = payload.get("workflow_run", {})
    conclusion = workflow_run.get("conclusion")

    # Only process completed runs with conclusion=failure
    if action != "completed" or conclusion != "failure":
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "ignored",
                "reason": f"Workflow run status is {action}/{conclusion}; only completed failures are triaged.",
            },
        )

    # 3. Extract Failing Context
    repo_info = payload.get("repository", {})
    repo_full_name = repo_info.get("full_name", "")
    branch = workflow_run.get("head_branch", "main")
    commit_sha = workflow_run.get("head_sha", "")
    run_id = workflow_run.get("id", 0)
    workflow_name = workflow_run.get("name", "CI Workflow")

    # Extract failing job details from check_suite or jobs if provided, or default
    failing_job_data = FailingJob(
        job_id=run_id,
        job_name=workflow_name,
        step_name=None,
        conclusion=conclusion or "failure",
        html_url=workflow_run.get("html_url"),
        annotations=[],
    )

    try:
        event = WorkflowFailureEvent(
            repo=repo_full_name,
            branch=branch,
            commit_sha=commit_sha,
            run_id=run_id,
            workflow_name=workflow_name,
            failing_job=failing_job_data,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse failure event: {exc}",
        )

    # 4. Schedule background healing task
    background_tasks.add_task(run_healing_pipeline, event)

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "status": "accepted",
            "message": "Self-healing pipeline initiated in background",
            "event_id": str(event.event_id),
            "repo": event.repo,
            "run_id": event.run_id,
        },
    )


def cli() -> None:
    """CLI entrypoint configured in pyproject.toml."""
    import uvicorn
    cfg = get_settings()
    uvicorn.run("pipelinedoctor.api.main:app", host=cfg.API_HOST, port=cfg.API_PORT, reload=False)
