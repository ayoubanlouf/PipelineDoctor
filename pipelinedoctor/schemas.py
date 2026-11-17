"""
pipelinedoctor/schemas.py
=========================
Core Pydantic v2 contracts shared across the entire PipelineDoctor stack.

Design rules
------------
* Every schema is immutable (model_config frozen=True) so LangGraph nodes
  cannot accidentally mutate shared state references.
* All Literals / Enums are defined here and imported elsewhere — no inline
  strings scattered across modules.
* AgentState uses a TypedDict (LangGraph convention) so it can be used
  directly as the graph's state type annotation without wrapping.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FailureType(StrEnum):
    """Classification produced by the Triage agent node."""

    SELECTOR_DRIFT = "selector_drift"
    """Playwright / Selenium selector no longer matches the DOM."""

    APP_BUG = "app_bug"
    """Regression in application code (logic error, missing import, etc.)."""

    FLAKY_ENV = "flaky_env"
    """Non-deterministic environment failure (timing, network, resource)."""


class AgentPhase(StrEnum):
    """LangGraph node names — single source of truth for routing keys."""

    TRIAGE = "triage"
    PATCH = "patch"
    VERIFY = "verify"
    DONE = "done"
    ESCALATE = "escalate"  # emitted when max iterations are exhausted


class VerificationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SANDBOX_ERROR = "sandbox_error"


# ---------------------------------------------------------------------------
# Webhook / Ingress contract
# ---------------------------------------------------------------------------


class FailingJob(BaseModel):
    """Subset of a GitHub Actions job object for a failed step."""

    model_config = {"frozen": True}

    job_id: int = Field(..., description="GitHub Actions job ID")
    job_name: str = Field(..., description="Human-readable job name")
    step_name: str | None = Field(None, description="Failing step name, if determinable")
    conclusion: str = Field(
        ...,
        description="GitHub conclusion string (failure | cancelled | timed_out | ...)",
    )
    html_url: HttpUrl | None = Field(None, description="Deep link to the failed job in GitHub UI")
    annotations: list[str] = Field(
        default_factory=list,
        description="Check-run annotation messages surfaced by GitHub",
    )


class WorkflowFailureEvent(BaseModel):
    """
    Payload emitted by GitHub Actions ``workflow_run`` webhook on failure.

    PipelineDoctor's FastAPI ingress deserialises the raw GitHub JSON into
    this schema before handing it to the agent graph.
    """

    model_config = {"frozen": True}

    # ── Identity ──────────────────────────────────────────────────────────
    event_id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Internal correlation ID assigned at ingestion time",
    )
    received_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of webhook receipt",
    )

    # ── Repository context ────────────────────────────────────────────────
    repo: str = Field(
        ...,
        examples=["acme-org/frontend-app"],
        description="GitHub repository in ``owner/repo`` format",
    )
    branch: str = Field(
        ...,
        examples=["main", "feature/checkout-redesign"],
        description="Git branch on which the workflow ran",
    )
    commit_sha: str = Field(
        ...,
        min_length=40,
        max_length=40,
        description="Full 40-character Git commit SHA",
    )

    # ── Workflow context ───────────────────────────────────────────────────
    run_id: int = Field(..., description="GitHub Actions ``workflow_run.id``")
    workflow_name: str = Field(
        default="CI Workflow",
        description="Name of the failed GitHub Actions workflow",
    )
    failing_job: FailingJob | None = Field(
        default=None,
        description="The specific job that failed",
    )

    # ── Raw artefacts (populated lazily by the integrations layer) ────────
    raw_logs: str | None = Field(
        None,
        description="Full text of the failing job's log, fetched from GitHub",
    )
    playwright_artifact_url: HttpUrl | None = Field(
        None,
        description="URL of the Playwright HTML report artefact, if applicable",
    )

    @model_validator(mode="after")
    def _ensure_failing_job(self) -> WorkflowFailureEvent:
        if self.failing_job is None:
            object.__setattr__(
                self,
                "failing_job",
                FailingJob(
                    job_id=self.run_id,
                    job_name=self.workflow_name,
                    conclusion="failure",
                ),
            )
        return self

    @field_validator("commit_sha")
    @classmethod
    def _validate_sha(cls, v: str) -> str:
        if not all(c in "0123456789abcdefABCDEF" for c in v):
            raise ValueError(f"commit_sha must be a valid hex string, got: {v!r}")
        return v.lower()

    @field_validator("repo")
    @classmethod
    def _validate_repo(cls, v: str) -> str:
        if "/" not in v or v.count("/") != 1:
            raise ValueError("repo must be in 'owner/repo' format")
        return v


# ---------------------------------------------------------------------------
# Triage Agent output
# ---------------------------------------------------------------------------


class TriageReport(BaseModel):
    """
    Structured output produced by the **Triage** LangGraph node.

    The LLM is prompted to return JSON matching this schema.  The smoke-test
    script (``scripts/test_llm_contract.py``) validates that Qwen3.5 9B can
    reliably produce valid instances of this class.
    """

    model_config = {"frozen": True}

    # ── Classification ────────────────────────────────────────────────────
    failure_type: FailureType = Field(
        ...,
        description="Coarse failure category used to route the Patch node's strategy",
    )

    # ── Root-cause analysis ───────────────────────────────────────────────
    root_cause: str = Field(
        ...,
        min_length=20,
        max_length=2000,
        description=(
            "1–3 sentence plain-English explanation of the failure root cause, "
            "grounded in evidence from the logs / artefacts"
        ),
    )

    # ── File targeting ────────────────────────────────────────────────────
    target_files: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Relative paths (from repo root) of files the Patch node should inspect. "
            "Ordered by relevance descending."
        ),
        examples=[["tests/e2e/checkout.spec.ts", "src/components/CheckoutButton.tsx"]],
    )

    # ── Confidence & evidence ─────────────────────────────────────────────
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="LLM self-reported confidence in the classification (0–1)",
    )
    evidence_snippets: list[str] = Field(
        default_factory=list,
        description="Verbatim log / stack-trace lines that support the root_cause",
    )

    @model_validator(mode="after")
    def _at_least_one_target(self) -> "TriageReport":
        if not self.target_files:
            raise ValueError("target_files must contain at least one path")
        return self


# ---------------------------------------------------------------------------
# Patch Agent output
# ---------------------------------------------------------------------------


class PatchProposal(BaseModel):
    """
    A single file-level patch proposed by the **Patch** LangGraph node.

    One ``PatchProposal`` per file; the node may return multiple proposals
    bundled in a list inside ``AgentState.patch_proposals``.
    """

    model_config = {"frozen": True}

    file_path: str = Field(
        ...,
        description="Repo-root-relative path of the file being patched",
        examples=["tests/e2e/checkout.spec.ts"],
    )
    unified_diff: str = Field(
        ...,
        description=(
            "Complete unified diff (``diff -u`` format) for this file. "
            "Must include ``--- a/`` and ``+++ b/`` headers."
        ),
    )
    explanation: str = Field(
        ...,
        min_length=20,
        max_length=1000,
        description="Concise rationale for this patch, written for a PR reviewer",
    )
    patch_strategy: str = Field(
        default="direct_edit",
        description=(
            "High-level strategy used (e.g. 'selector_update', 'retry_wrapper', "
            "'env_pin', 'logic_fix')"
        ),
    )

    @field_validator("unified_diff")
    @classmethod
    def _validate_diff(cls, v: str) -> str:
        if "---" not in v or "+++" not in v:
            raise ValueError(
                "unified_diff must contain '---' and '+++' headers (unified diff format)"
            )
        return v


# ---------------------------------------------------------------------------
# Verification result (produced by Sandbox node)
# ---------------------------------------------------------------------------


class VerificationResult(BaseModel):
    """Outcome of running the patched tests inside an ephemeral Docker sandbox."""

    model_config = {"frozen": True}

    status: VerificationStatus
    exit_code: int = Field(..., ge=-1, description="-1 indicates a sandbox/timeout failure")
    stdout: str = Field(default="", description="Captured stdout from the test runner")
    stderr: str = Field(default="", description="Captured stderr from the test runner")
    duration_seconds: float = Field(default=0.0, ge=0.0)
    container_id: str | None = Field(None, description="Docker container ID for audit trail")

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASSED


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """
    Mutable state bag threaded through every LangGraph node.

    Using ``TypedDict`` (not a Pydantic model) because LangGraph's
    ``StateGraph`` expects a plain TypedDict as its state annotation.
    All *mutable* fields use ``total=False`` so nodes can safely do
    partial updates via ``Command(update={...})``.

    Immutable seed values (``event``, ``run_id``) are always present;
    they are set once at graph entry and never overwritten.
    """

    # ── Seed (always present) ─────────────────────────────────────────────
    event: WorkflowFailureEvent
    """The original webhook payload that triggered the heal cycle."""

    # ── Routing & lifecycle ───────────────────────────────────────────────
    current_phase: AgentPhase
    """Which node is currently executing (for debugging / observability)."""

    iteration: int
    """Current heal iteration (1-indexed). Graph aborts when this exceeds MAX_ITERATIONS."""

    max_iterations: int
    """Hard cap on heal loops (default: 3). Injected at graph construction time."""

    # ── Triage node outputs ───────────────────────────────────────────────
    triage_report: TriageReport | None
    """Populated by the Triage node; ``None`` before first triage."""

    # ── Patch node outputs ────────────────────────────────────────────────
    patch_proposals: list[PatchProposal]
    """All patches proposed across all iterations."""

    applied_diffs: list[str]
    """Unified diffs that were actually written to the sandbox container."""

    # ── Verification node outputs ─────────────────────────────────────────
    verification_results: list[VerificationResult]
    """One entry per verify attempt, ordered chronologically."""

    latest_verification: VerificationResult | None
    """Convenience alias — always the last element of verification_results."""

    # ── Raw artefacts & logs ──────────────────────────────────────────────
    fetched_logs: str
    """Raw CI log text fetched from GitHub (or truncated to 32 KB)."""

    fetched_file_contents: dict[str, str]
    """Mapping of ``file_path → raw source`` for target files read from the repo."""

    local_repo_path: str | None
    """Optional path to a local directory snapshot of the repository for verification."""

    # ── GitHub integration ────────────────────────────────────────────────
    fix_branch: str | None
    """Name of the git branch created to hold the fix."""

    pr_url: str | None
    """GitHub PR URL, set after the draft PR is opened on success."""

    # ── Telemetry ─────────────────────────────────────────────────────────
    llm_call_count: int
    """Total number of LLM API calls made during this heal cycle."""

    token_usage: dict[str, Any]
    """Accumulated token usage reported by the OpenAI-compat endpoint."""

    errors: list[str]
    """Non-fatal error messages accumulated across nodes (for the escalation report)."""
