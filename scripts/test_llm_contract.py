#!/usr/bin/env python3
"""
scripts/test_llm_contract.py
=============================
Standalone smoke-test: validates that the local Bionic/GGUF endpoint at
http://localhost:7800/v1 (model: Qwen3.5 9B) reliably produces valid JSON
matching **TriageReport** and **PatchProposal**.

Execution
---------
    python scripts/test_llm_contract.py              # normal run
    python scripts/test_llm_contract.py --verbose    # print raw LLM responses
    python scripts/test_llm_contract.py --rounds 5   # repeat each test N times

Design decisions
----------------
1. Uses the **openai** SDK pointed at the local base_url — zero infra overhead.
2. Attempts native ``response_format={"type": "json_schema"}`` first (OpenAI
   structured-outputs). If the server returns 400 / 422 or the payload is not
   valid JSON, it transparently falls back to regex + ``json.loads`` extraction.
3. Pydantic v2 ``.model_validate()`` is the final gate — the test only passes
   when the parsed dict satisfies all field constraints and validators.
4. Rich terminal output with PASS / FAIL / WARN labels so CI can grep for
   "FAILED" in stdout and exit non-zero when things are broken.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

# ---------------------------------------------------------------------------
# Ensure the project root is on PYTHONPATH so we can import pipelinedoctor
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from openai import OpenAI, OpenAIError  # noqa: E402
from pydantic import BaseModel, ValidationError  # noqa: E402

from pipelinedoctor.schemas import (  # noqa: E402
    PatchProposal,
    TriageReport,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Default Bionic endpoint and model ID
LLM_BASE_URL = "http://localhost:1235/v1"
LLM_API_KEY = "not-needed"           # Bionic ignores Authorization header
LLM_MODEL = "qwen3.5-9b-uncensored-hauhaucs-aggressive"  # Exact model ID from /v1/models
LLM_TIMEOUT = 120                    # seconds per call
MAX_TOKENS = 2000                    # must be large enough for reasoning + JSON output

# ANSI colours (disabled when not a tty)
_IS_TTY = sys.stdout.isatty()
GREEN  = "\033[92m" if _IS_TTY else ""
YELLOW = "\033[93m" if _IS_TTY else ""
RED    = "\033[91m" if _IS_TTY else ""
BOLD   = "\033[1m"  if _IS_TTY else ""
RESET  = "\033[0m"  if _IS_TTY else ""

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(msg: str) -> None:
    print(f"  {GREEN}✔ PASS{RESET}  {msg}")

def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠ WARN{RESET}  {msg}")

def _fail(msg: str) -> None:
    print(f"  {RED}✘ FAIL{RESET}  {msg}")

def _header(msg: str) -> None:
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  {msg}{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")


def _extract_json_fallback(text: str) -> dict[str, Any]:
    """
    Regex-based fallback parser.

    Strategies (in order):
    1. Find the first ``{...}`` block (possibly multi-line) using a greedy scan.
    2. Strip markdown code-fence wrappers (```json ... ```).
    3. Try ``json.loads`` on the raw text directly.
    """
    # Strategy 1 – strip code fences then parse
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 2 – find first balanced brace block
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    # Strategy 3 – raw parse
    return json.loads(text)


def _strip_think_tags(text: str) -> str:
    """
    Remove Qwen3-style <think>...</think> reasoning blocks from content.

    Bionic normally hides these in `content` but exposes token counts in
    `completion_tokens_details.reasoning_tokens`.  Other GGUF frontends
    (e.g. llama.cpp, LM Studio) may leave the raw tags in the content string.
    This guard makes the parser robust to both behaviours.
    """
    # Greedy strip — multiple think blocks possible
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    return cleaned.strip()


def _call_llm(
    client: OpenAI,
    prompt: str,
    schema: type[T],
    *,
    model: str = LLM_MODEL,
    use_structured_output: bool = True,
    verbose: bool = False,
) -> tuple[T, str, bool]:
    """
    Call the LLM and return ``(parsed_model, raw_text, used_structured_output)``.

    ``used_structured_output=False`` means we fell back to regex parsing.
    """
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert CI/CD automated repair assistant.\n"
                "You must analyze the problem and output ONLY a valid JSON object matching this JSON schema:\n"
                f"{schema_json}\n\n"
                "Rules:\n"
                "- Output ONLY the JSON object. Do not wrap in markdown or backticks.\n"
                "- Ensure all required schema fields are present and follow their types and constraints.\n"
                "- Explanations and root causes must be full, descriptive sentences (at least 20 characters).\n"
            ),
        },
        {"role": "user", "content": prompt},
    ]

    raw_text: str = ""
    used_so = use_structured_output

    # ── Attempt 1: native structured output ──────────────────────────────
    if use_structured_output:
        try:
            json_schema = schema.model_json_schema()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=MAX_TOKENS,
                timeout=LLM_TIMEOUT,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": json_schema,
                        "strict": False,
                    },
                },
            )
            raw_text = resp.choices[0].message.content or ""
            raw_text = _strip_think_tags(raw_text)
            if verbose:
                print(f"\n    [RAW structured-output response]\n{textwrap.indent(raw_text, '    ')}")
            if not raw_text.strip():
                raise ValueError("Structured output returned empty content (token budget exhausted by reasoning).")
            parsed_dict = json.loads(raw_text)
            return schema.model_validate(parsed_dict), raw_text, True
        except (OpenAIError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            _warn(f"Structured-output attempt failed ({type(exc).__name__}: {exc}). Falling back.")
            used_so = False

    # ── Attempt 2: plain completion + fallback parser ─────────────────────
    if not use_structured_output or not used_so:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=MAX_TOKENS,
            timeout=LLM_TIMEOUT,
        )
        raw_text = resp.choices[0].message.content or ""
        # Strip think tags BEFORE parsing (safety net for GGUF frontends)
        raw_text = _strip_think_tags(raw_text)
        if verbose:
            print(f"\n    [RAW fallback response]\n{textwrap.indent(raw_text, '    ')}")

    parsed_dict = _extract_json_fallback(raw_text)
    return schema.model_validate(parsed_dict), raw_text, used_so


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

TRIAGE_PROMPT = textwrap.dedent("""\
    A GitHub Actions workflow run failed. Here is the relevant log snippet:

    ```
    FAIL tests/e2e/checkout.spec.ts
      ● Checkout flow › submits order successfully
        Error: locator.click: Error: strict mode violation:
          locator('#checkout-btn') resolved to 3 elements.

    at tests/e2e/checkout.spec.ts:42:5
    ```

    Classify this failure and return a JSON object with:
    - failure_type: one of "selector_drift", "app_bug", "flaky_env"
    - root_cause: 1-3 sentence plain English explanation
    - target_files: list of file paths that need to be fixed (relative to repo root)
    - confidence: float 0.0-1.0
    - evidence_snippets: list of relevant log lines
""")

PATCH_PROMPT = textwrap.dedent("""\
    The Playwright test `tests/e2e/checkout.spec.ts` uses the CSS selector `#checkout-btn`
    which now resolves to 3 elements after a UI refactor. The correct new selector is
    `[data-testid="checkout-submit-btn"]`.

    Return a JSON object with a patch proposal:
    - file_path: the file being patched
    - unified_diff: a valid unified diff (--- a/ +++ b/ format) replacing the old selector
    - explanation: concise rationale for the change (for a PR reviewer)
    - patch_strategy: high-level strategy (e.g. "selector_update")
""")


@dataclass
class TestResult:
    name: str
    passed: bool
    used_structured_output: bool
    duration_seconds: float
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual test runners
# ---------------------------------------------------------------------------

def test_triage_report(
    client: OpenAI,
    *,
    model: str = LLM_MODEL,
    verbose: bool = False,
    use_structured_output: bool = True,
) -> TestResult:
    """Verify that the LLM produces a valid TriageReport."""
    _header("TEST: TriageReport structured output")
    t0 = time.monotonic()
    warnings: list[str] = []

    try:
        result, raw, used_so = _call_llm(
            client, TRIAGE_PROMPT, TriageReport,
            model=model,
            use_structured_output=use_structured_output,
            verbose=verbose,
        )
        duration = time.monotonic() - t0

        # Semantic checks (beyond Pydantic validation)
        if result.failure_type.value != "selector_drift":
            warnings.append(
                f"Expected failure_type='selector_drift', got '{result.failure_type}' "
                f"(model classified differently — not a hard failure)"
            )
        if result.confidence < 0.3:
            warnings.append(f"confidence={result.confidence:.2f} is very low")
        if not result.target_files:
            warnings.append("target_files is empty")

        for w in warnings:
            _warn(w)

        _ok(f"TriageReport valid  (failure_type={result.failure_type!r}, "
            f"confidence={result.confidence:.2f}, "
            f"files={result.target_files}, "
            f"{'structured-output' if used_so else 'regex-fallback'}, "
            f"{duration:.2f}s)")

        return TestResult(
            name="triage_report",
            passed=True,
            used_structured_output=used_so,
            duration_seconds=duration,
            warnings=warnings,
        )

    except (ValidationError, json.JSONDecodeError, OpenAIError, ValueError) as exc:
        duration = time.monotonic() - t0
        _fail(f"TriageReport FAILED: {exc}")
        return TestResult(
            name="triage_report",
            passed=False,
            used_structured_output=False,
            duration_seconds=duration,
            error=str(exc),
        )


def test_patch_proposal(
    client: OpenAI,
    *,
    model: str = LLM_MODEL,
    verbose: bool = False,
    use_structured_output: bool = True,
) -> TestResult:
    """Verify that the LLM produces a valid PatchProposal with a well-formed diff."""
    _header("TEST: PatchProposal structured output")
    t0 = time.monotonic()
    warnings: list[str] = []

    try:
        result, raw, used_so = _call_llm(
            client, PATCH_PROMPT, PatchProposal,
            model=model,
            use_structured_output=use_structured_output,
            verbose=verbose,
        )
        duration = time.monotonic() - t0

        # Semantic diff checks
        if "data-testid" not in result.unified_diff:
            warnings.append(
                "unified_diff does not mention 'data-testid' — patch may be incomplete"
            )
        diff_lines = result.unified_diff.splitlines()
        has_add = any(l.startswith("+") and not l.startswith("+++") for l in diff_lines)
        has_remove = any(l.startswith("-") and not l.startswith("---") for l in diff_lines)
        if not (has_add and has_remove):
            warnings.append("unified_diff has no +/- change lines — may be a no-op patch")

        for w in warnings:
            _warn(w)

        _ok(f"PatchProposal valid  (file={result.file_path!r}, "
            f"strategy={result.patch_strategy!r}, "
            f"{'structured-output' if used_so else 'regex-fallback'}, "
            f"{duration:.2f}s)")

        return TestResult(
            name="patch_proposal",
            passed=True,
            used_structured_output=used_so,
            duration_seconds=duration,
            warnings=warnings,
        )

    except (ValidationError, json.JSONDecodeError, OpenAIError, ValueError) as exc:
        duration = time.monotonic() - t0
        _fail(f"PatchProposal FAILED: {exc}")
        return TestResult(
            name="patch_proposal",
            passed=False,
            used_structured_output=False,
            duration_seconds=duration,
            error=str(exc),
        )


def test_connectivity(client: OpenAI) -> bool:
    """Ping the endpoint and verify the model is listed / reachable."""
    _header("TEST: Endpoint connectivity")
    try:
        models = client.models.list()
        names = [m.id for m in models.data]
        if names:
            _ok(f"Endpoint reachable at {client.base_url}. Available models: {names}")
        else:
            _warn(f"Endpoint reachable at {client.base_url} but returned an empty model list")
        return True
    except OpenAIError as exc:
        _fail(f"Cannot reach {client.base_url}: {exc}")
        print(
            f"\n  {YELLOW}Hint:{RESET} Make sure your Bionic (or llama.cpp) server is running at {client.base_url}\n"
        )
        return False


# ---------------------------------------------------------------------------
# Summary reporter
# ---------------------------------------------------------------------------

def _print_summary(results: list[TestResult]) -> int:
    """Print a human-readable summary table. Returns non-zero if any test failed."""
    _header("SUMMARY")
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    for r in results:
        icon = f"{GREEN}✔{RESET}" if r.passed else f"{RED}✘{RESET}"
        so_tag = "native-SO" if r.used_structured_output else "regex-fallback"
        warn_tag = f"  [{YELLOW}{len(r.warnings)} warn(s){RESET}]" if r.warnings else ""
        dur = f"{r.duration_seconds:.2f}s"
        print(f"  {icon}  {r.name:<20}  {so_tag:<14}  {dur}{warn_tag}")
        if r.error:
            print(f"       └─ error: {RED}{r.error}{RESET}")

    print()
    color = GREEN if failed == 0 else RED
    print(f"  {color}{BOLD}{passed}/{total} tests passed{RESET}")

    # Capability assessment
    all_native_so = all(r.used_structured_output for r in results if r.passed)
    if all_native_so:
        print(
            f"\n  {GREEN}Model supports native structured output.{RESET}\n"
            "  → LangGraph nodes can use ``with_structured_output()`` directly."
        )
    else:
        print(
            f"\n  {YELLOW}Model does NOT support native structured output.{RESET}\n"
            "  → LangGraph nodes should use the regex/JSON fallback parser.\n"
            "  → Consider adding an output parser middleware in pipelinedoctor/agents/."
        )

    return 1 if failed > 0 else 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test: validate Qwen3.5 9B structured output for PipelineDoctor schemas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
              python scripts/test_llm_contract.py
              python scripts/test_llm_contract.py --verbose
              python scripts/test_llm_contract.py --no-structured-output   # force fallback parser
              python scripts/test_llm_contract.py --rounds 3               # run each test 3 times
        """),
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print raw LLM response text")
    parser.add_argument("--rounds", "-n", type=int, default=1,
                        help="Repeat each contract test N times (default: 1)")
    parser.add_argument("--no-structured-output", dest="use_so", action="store_false",
                        default=True, help="Skip native structured-output attempt, go straight to fallback")
    parser.add_argument("--base-url", default=LLM_BASE_URL,
                        help=f"LLM base URL (default: {LLM_BASE_URL})")
    parser.add_argument("--model", default=LLM_MODEL,
                        help=f"Model identifier (default: {LLM_MODEL!r})")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=LLM_API_KEY)

    print(f"\n{BOLD}PipelineDoctor — LLM Contract Smoke Test{RESET}")
    print(f"  Endpoint : {args.base_url}")
    print(f"  Model    : {args.model}")
    print(f"  Rounds   : {args.rounds}")
    print(f"  Native SO: {args.use_so}")

    # 1. Connectivity gate — abort early if the server isn't up
    if not test_connectivity(client):
        return 1

    # 2. Contract tests (optionally repeated for consistency check)
    all_results: list[TestResult] = []

    for round_num in range(1, args.rounds + 1):
        if args.rounds > 1:
            print(f"\n{BOLD}═══ Round {round_num}/{args.rounds} ═══{RESET}")

        all_results.append(
            test_triage_report(
                client,
                model=args.model,
                verbose=args.verbose,
                use_structured_output=args.use_so,
            )
        )
        all_results.append(
            test_patch_proposal(
                client,
                model=args.model,
                verbose=args.verbose,
                use_structured_output=args.use_so,
            )
        )

    return _print_summary(all_results)


if __name__ == "__main__":
    sys.exit(main())
