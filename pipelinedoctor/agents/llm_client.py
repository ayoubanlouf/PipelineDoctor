"""
pipelinedoctor/agents/llm_client.py
==================================
OpenAI-compatible client wrapper with reasoning tag stripping and
resilient JSON schema fallback extraction for TriageReport and PatchProposal.
"""

from __future__ import annotations

import json
import re
import textwrap
from typing import Any, TypeVar

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from pipelinedoctor.config import Settings, get_settings
from pipelinedoctor.schemas import PatchProposal, TriageReport

T = TypeVar("T", bound=BaseModel)


def strip_think_tags(text: str) -> str:
    """
    Remove reasoning blocks (e.g., <think>...</think>) from model output.
    Supports multiline and case-insensitive tags.
    """
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    return cleaned.strip()


def extract_json(text: str) -> dict[str, Any]:
    """
    Robust fallback JSON extraction strategy:
    1. Check for markdown code fences (```json ... ```).
    2. Balanced brace scan to find the first complete JSON object.
    3. Direct raw json.loads.
    """
    # 1. Fenced block
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if fenced_match:
        try:
            return json.loads(fenced_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 2. Balanced brace extraction
    start_idx = text.find("{")
    if start_idx != -1:
        depth = 0
        for idx in range(start_idx, len(text)):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx : idx + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    # 3. Direct parse
    return json.loads(text.strip())


class LLMClient:
    """High-level client for PipelineDoctor LLM operations."""

    def __init__(self, settings: Settings | None = None, client: OpenAI | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or OpenAI(
            base_url=self.settings.BIONIC_BASE_URL,
            api_key="not-needed",
            timeout=float(self.settings.LLM_TIMEOUT_SECONDS),
        )

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str | None = None,
        use_native_so: bool = True,
    ) -> tuple[T, dict[str, Any]]:
        """
        Request structured output from the LLM, returning (model_instance, token_usage).
        Attempts native json_schema first, falling back to plain prompt + regex parser.
        """
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        base_system = (
            system_prompt
            or (
                "You are an expert autonomous CI/CD and QA self-healing system.\n"
                "You must analyze the failure and respond ONLY with a valid JSON object matching this schema:\n"
                f"{schema_json}\n\n"
                "Rules:\n"
                "- Output ONLY a single JSON object. No markdown wrappers or explanation outside the JSON.\n"
                "- All root causes and explanations must be full descriptive sentences (at least 20 characters).\n"
                "- Unified diffs must include valid '--- a/' and '+++ b/' headers."
            )
        )

        messages = [
            {"role": "system", "content": base_system},
            {"role": "user", "content": prompt},
        ]

        usage_info: dict[str, Any] = {}
        raw_text = ""

        # Attempt 1: Native json_schema
        if use_native_so:
            try:
                resp = self.client.chat.completions.create(
                    model=self.settings.BIONIC_MODEL,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=self.settings.LLM_MAX_TOKENS,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema.__name__,
                            "schema": schema.model_json_schema(),
                            "strict": False,
                        },
                    },
                )
                if resp.usage:
                    usage_info = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                    }
                content = resp.choices[0].message.content or ""
                raw_text = strip_think_tags(content)
                if raw_text:
                    parsed = json.loads(raw_text)
                    return schema.model_validate(parsed), usage_info
            except (OpenAIError, json.JSONDecodeError, ValidationError, ValueError, Exception):
                # Fall back to Attempt 2
                pass

        # Attempt 2: Plain completion + robust extraction
        resp = self.client.chat.completions.create(
            model=self.settings.BIONIC_MODEL,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=self.settings.LLM_MAX_TOKENS,
        )
        if resp.usage:
            usage_info = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        content = resp.choices[0].message.content or ""
        raw_text = strip_think_tags(content)
        parsed_dict = extract_json(raw_text)
        return schema.model_validate(parsed_dict), usage_info

    def triage_failure(
        self,
        job_name: str,
        logs: str,
        annotations: list[str] | None = None,
    ) -> tuple[TriageReport, dict[str, Any]]:
        """Prompt the LLM to classify failure and pinpoint target files."""
        # Truncate logs if overly large to fit context window
        snippet = logs[-8000:] if len(logs) > 8000 else logs
        annotation_block = "\n".join(f"- {a}" for a in (annotations or []))

        prompt = textwrap.dedent(f"""\
            A CI/CD workflow job failed. Analyze the logs and triage the failure.

            Failed Job: {job_name}
            Annotations:
            {annotation_block or "None"}

            Recent Logs Snippet:
            ```
            {snippet}
            ```

            Identify the failure_type (selector_drift | app_bug | flaky_env), provide a detailed root_cause (>=20 chars),
            and list the target_files (relative paths) that should be inspected or patched.
        """)

        return self.generate_structured(prompt=prompt, schema=TriageReport)

    def propose_patch(
        self,
        file_path: str,
        current_content: str,
        triage: TriageReport,
        previous_diffs: list[str] | None = None,
        verification_error: str | None = None,
    ) -> tuple[PatchProposal, dict[str, Any]]:
        """Prompt the LLM to generate a unified diff patch for a target file."""
        history_context = ""
        if previous_diffs:
            history_context += "\nPreviously attempted patches that failed verification:\n"
            for d in previous_diffs:
                history_context += f"```diff\n{d}\n```\n"

        if verification_error:
            history_context += f"\nSandbox verification error:\n```\n{verification_error}\n```\n"

        prompt = textwrap.dedent(f"""\
            Generate a targeted bugfix patch for the file '{file_path}'.

            Triage Analysis:
            - Failure Type: {triage.failure_type.value}
            - Root Cause: {triage.root_cause}

            Current File Content:
            ```{file_path}
            {current_content}
            ```
            {history_context}
            Generate a valid unified diff (with '--- a/{file_path}' and '+++ b/{file_path}' headers),
            a clear explanation (>=20 chars), and the patch_strategy (e.g. 'selector_update', 'logic_fix').
        """)

        return self.generate_structured(prompt=prompt, schema=PatchProposal)
