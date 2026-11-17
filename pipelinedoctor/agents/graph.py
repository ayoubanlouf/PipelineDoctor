"""
pipelinedoctor/agents/graph.py
==============================
LangGraph state machine for autonomous CI/CD self-healing.
Enforces a bounded Triage -> Patch -> Verify loop (max 3 iterations),
with deterministic verification inside ephemeral Docker containers and
escalation upon iteration exhaustion.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from pipelinedoctor.agents.llm_client import LLMClient
from pipelinedoctor.config import Settings, get_settings
from pipelinedoctor.integrations.github import GitHubIntegration
from pipelinedoctor.sandbox.executor import SandboxExecutor
from pipelinedoctor.schemas import (
    AgentPhase,
    AgentState,
    PatchProposal,
    TriageReport,
    VerificationResult,
    VerificationStatus,
)


def create_healing_graph(
    llm_client: LLMClient | None = None,
    sandbox_executor: SandboxExecutor | None = None,
    github_integration: GitHubIntegration | None = None,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """Factory creating the LangGraph compiled state machine with injected dependencies."""
    cfg = settings or get_settings()
    llm = llm_client or LLMClient(settings=cfg)
    sandbox = sandbox_executor or SandboxExecutor(settings=cfg)
    gh = github_integration or GitHubIntegration(settings=cfg)

    # ───────────────────────────────────────────────────────────────────────
    # Node 1: Triage
    # ───────────────────────────────────────────────────────────────────────
    def triage_node(state: AgentState) -> dict[str, Any]:
        event = state["event"]
        logs = state.get("fetched_logs") or ""

        # Fetch CI logs if not present in state
        if not logs:
            logs = gh.fetch_workflow_logs(
                repo_name=event.repo,
                run_id=event.run_id,
                job_id=event.failing_job.job_id,
            )

        triage_report, usage = llm.triage_failure(
            job_name=event.failing_job.job_name,
            logs=logs,
            annotations=event.failing_job.annotations,
        )

        # Pre-fetch target file contents
        file_contents = dict(state.get("fetched_file_contents", {}))
        for target_file in triage_report.target_files:
            if target_file not in file_contents:
                try:
                    content = gh.fetch_file_content(
                        repo_name=event.repo,
                        file_path=target_file,
                        ref=event.commit_sha,
                    )
                    file_contents[target_file] = content
                except Exception:
                    file_contents[target_file] = ""

        call_count = state.get("llm_call_count", 0) + 1

        return {
            "current_phase": AgentPhase.TRIAGE,
            "fetched_logs": logs,
            "triage_report": triage_report,
            "fetched_file_contents": file_contents,
            "llm_call_count": call_count,
            "token_usage": usage,
        }

    # ───────────────────────────────────────────────────────────────────────
    # Node 2: Patch
    # ───────────────────────────────────────────────────────────────────────
    def patch_node(state: AgentState) -> dict[str, Any]:
        triage: TriageReport = state["triage_report"]
        file_contents = state.get("fetched_file_contents", {})
        existing_proposals = list(state.get("patch_proposals", []))
        existing_diffs = list(state.get("applied_diffs", []))
        latest_verif: VerificationResult | None = state.get("latest_verification")

        verif_err = latest_verif.stderr if latest_verif and not latest_verif.passed else None

        # Propose patch for primary target file
        target_file = triage.target_files[0]
        content = file_contents.get(target_file, "")

        proposal, usage = llm.propose_patch(
            file_path=target_file,
            current_content=content,
            triage=triage,
            previous_diffs=existing_diffs if existing_diffs else None,
            verification_error=verif_err,
        )

        new_proposals = existing_proposals + [proposal]
        new_diffs = existing_diffs + [proposal.unified_diff]
        call_count = state.get("llm_call_count", 0) + 1

        return {
            "current_phase": AgentPhase.PATCH,
            "patch_proposals": new_proposals,
            "applied_diffs": new_diffs,
            "llm_call_count": call_count,
        }

    # ───────────────────────────────────────────────────────────────────────
    # Node 3: Verify
    # ───────────────────────────────────────────────────────────────────────
    def verify_node(state: AgentState) -> dict[str, Any]:
        diffs = state.get("applied_diffs", [])
        current_iteration = state.get("iteration", 0) + 1
        results = list(state.get("verification_results", []))

        # Run verification inside isolated Docker sandbox
        # Target repo can be local path or clone path
        repo_path = state.get("local_repo_path") or state.get("event").repo
        res = sandbox.execute_in_sandbox(
            repo_dir=repo_path,
            diffs=diffs,
            test_command="pytest",
        )

        results.append(res)

        return {
            "current_phase": AgentPhase.VERIFY,
            "iteration": current_iteration,
            "verification_results": results,
            "latest_verification": res,
        }

    # ───────────────────────────────────────────────────────────────────────
    # Routing condition
    # ───────────────────────────────────────────────────────────────────────
    def should_continue(state: AgentState) -> Literal["done", "escalate", "patch"]:
        latest: VerificationResult | None = state.get("latest_verification")
        iteration = state.get("iteration", 0)
        max_iters = state.get("max_iterations", cfg.MAX_ITERATIONS)

        if latest and latest.passed:
            return "done"

        if iteration >= max_iters:
            return "escalate"

        # Verification failed but iterations remaining -> retry patch
        return "patch"

    # ───────────────────────────────────────────────────────────────────────
    # Node 4: Done (Deterministic Verification Passed -> Push & Draft PR)
    # ───────────────────────────────────────────────────────────────────────
    def done_node(state: AgentState) -> dict[str, Any]:
        event = state["event"]
        proposals = state.get("patch_proposals", [])
        diffs = state.get("applied_diffs", [])
        triage = state.get("triage_report")

        branch_name = f"pipelinedoctor/fix-run-{event.run_id}"

        # Generate updated file contents
        # For simplicity in draft PR, updates map file_path -> new content or diff
        file_updates = {p.file_path: p.explanation for p in proposals}

        try:
            gh.create_fix_branch_and_commit(
                repo_name=event.repo,
                base_sha=event.commit_sha,
                branch_name=branch_name,
                file_updates=file_updates,
                commit_message=f"fix(ci): PipelineDoctor self-healing fix for run #{event.run_id}",
            )

            pr_body = (
                f"## 🤖 PipelineDoctor Autonomous Self-Healing Fix\n\n"
                f"- **Failed Run:** #{event.run_id} ({event.workflow_name})\n"
                f"- **Failure Type:** {triage.failure_type.value if triage else 'Unknown'}\n"
                f"- **Root Cause:** {triage.root_cause if triage else 'N/A'}\n\n"
                f"### Verification Status\n"
                f"✔ Ephemeral Docker Sandbox verification passed with exit code 0.\n\n"
                f"### Proposed Patches\n"
            )
            for p in proposals:
                pr_body += f"#### `{p.file_path}`\n{p.explanation}\n```diff\n{p.unified_diff}\n```\n\n"

            pr_url = gh.open_draft_pr(
                repo_name=event.repo,
                head_branch=branch_name,
                base_branch=event.branch,
                title=f"fix(ci): autonomous repair for {event.workflow_name} run #{event.run_id}",
                body=pr_body,
            )
        except Exception as exc:
            pr_url = f"pr-creation-failed: {exc}"

        return {
            "current_phase": AgentPhase.DONE,
            "fix_branch": branch_name,
            "pr_url": pr_url,
        }

    # ───────────────────────────────────────────────────────────────────────
    # Node 5: Escalate (Max iterations reached without passing)
    # ───────────────────────────────────────────────────────────────────────
    def escalate_node(state: AgentState) -> dict[str, Any]:
        errors = list(state.get("errors", []))
        iters = state.get("iteration", 0)
        max_iters = state.get("max_iterations", cfg.MAX_ITERATIONS)
        errors.append(
            f"Escalated to human reviewer: Self-healing loop reached {iters}/{max_iters} "
            f"iterations without passing sandbox verification."
        )

        return {
            "current_phase": AgentPhase.ESCALATE,
            "errors": errors,
        }

    # ───────────────────────────────────────────────────────────────────────
    # Assemble Graph
    # ───────────────────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    graph.add_node(AgentPhase.TRIAGE, triage_node)
    graph.add_node(AgentPhase.PATCH, patch_node)
    graph.add_node(AgentPhase.VERIFY, verify_node)
    graph.add_node(AgentPhase.DONE, done_node)
    graph.add_node(AgentPhase.ESCALATE, escalate_node)

    graph.add_edge(START, AgentPhase.TRIAGE)
    graph.add_edge(AgentPhase.TRIAGE, AgentPhase.PATCH)
    graph.add_edge(AgentPhase.PATCH, AgentPhase.VERIFY)

    graph.add_conditional_edges(
        AgentPhase.VERIFY,
        should_continue,
        {
            "done": AgentPhase.DONE,
            "escalate": AgentPhase.ESCALATE,
            "patch": AgentPhase.PATCH,
        },
    )

    graph.add_edge(AgentPhase.DONE, END)
    graph.add_edge(AgentPhase.ESCALATE, END)

    return graph.compile()
