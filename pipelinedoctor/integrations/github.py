"""
pipelinedoctor/integrations/github.py
====================================
PyGithub client wrapper for fetching failing workflow logs,
reading source files at specific commits, creating fix branches,
and opening draft Pull Requests upon deterministic verification pass.
"""

from __future__ import annotations

import io
import zipfile
from typing import Any

from github import Github, GithubException

from pipelinedoctor.config import Settings, get_settings


class GitHubIntegration:
    """Handles GitHub API interactions for PipelineDoctor."""

    def __init__(
        self,
        settings: Settings | None = None,
        github_client: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = github_client

    @property
    def client(self) -> Github:
        if self._client is None:
            token = self.settings.GITHUB_TOKEN
            self._client = Github(token if token else None)
        return self._client

    def fetch_workflow_logs(self, repo_name: str, run_id: int, job_id: int | None = None) -> str:
        """
        Fetches text logs for a workflow run or specific job.
        GitHub provides workflow run logs as a zip archive containing per-job text files.
        """
        try:
            repo = self.client.get_repo(repo_name)
            run = repo.get_workflow_run(run_id)

            # PyGithub provides run.logs_url (URL redirect to zip download)
            # or job.logs() if available
            if job_id:
                try:
                    job = repo.get_workflow_job(job_id)
                    logs_text = job.logs()
                    if logs_text:
                        return logs_text
                except Exception:
                    pass

            # Fallback to downloading workflow run logs archive
            import requests

            headers = {"Authorization": f"Bearer {self.settings.GITHUB_TOKEN}"} if self.settings.GITHUB_TOKEN else {}
            resp = requests.get(run.logs_url, headers=headers, timeout=30, allow_redirects=True)
            if resp.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
                    all_logs = []
                    for name in archive.namelist():
                        if name.endswith(".txt"):
                            content = archive.read(name).decode("utf-8", errors="replace")
                            all_logs.append(f"=== {name} ===\n{content}")
                    return "\n\n".join(all_logs)

            return f"Failed to fetch logs. HTTP status {resp.status_code}"

        except GithubException as g_exc:
            return f"GitHub API error fetching logs: {g_exc.data if hasattr(g_exc, 'data') else g_exc}"
        except Exception as exc:
            return f"Unexpected error fetching logs: {exc}"

    def fetch_file_content(self, repo_name: str, file_path: str, ref: str) -> str:
        """Reads file text from GitHub repository at a specific ref/commit_sha."""
        try:
            repo = self.client.get_repo(repo_name)
            file_content = repo.get_contents(file_path, ref=ref)
            if hasattr(file_content, "decoded_content"):
                return file_content.decoded_content.decode("utf-8", errors="replace")
            return ""
        except GithubException as g_exc:
            raise FileNotFoundError(f"File '{file_path}' not found at ref '{ref}': {g_exc}")

    def create_fix_branch_and_commit(
        self,
        repo_name: str,
        base_sha: str,
        branch_name: str,
        file_updates: dict[str, str],
        commit_message: str,
    ) -> str:
        """
        Creates a new git branch from base_sha and applies file updates.
        Returns the new branch ref or latest commit sha.
        """
        repo = self.client.get_repo(repo_name)
        ref_path = f"refs/heads/{branch_name}"

        # 1. Create git branch
        try:
            repo.create_git_ref(ref=ref_path, sha=base_sha)
        except GithubException:
            # Branch may already exist from previous iteration
            pass

        # 2. Update files on the branch
        for path, new_content in file_updates.items():
            try:
                existing = repo.get_contents(path, ref=branch_name)
                repo.update_file(
                    path=path,
                    message=commit_message,
                    content=new_content,
                    sha=existing.sha,
                    branch=branch_name,
                )
            except GithubException:
                repo.create_file(
                    path=path,
                    message=commit_message,
                    content=new_content,
                    branch=branch_name,
                )

        return branch_name

    def open_draft_pr(
        self,
        repo_name: str,
        head_branch: str,
        base_branch: str,
        title: str,
        body: str,
    ) -> str:
        """Opens a Draft Pull Request and returns its HTML URL."""
        repo = self.client.get_repo(repo_name)
        pr = repo.create_pull(
            title=title,
            body=body,
            head=head_branch,
            base=base_branch,
            draft=True,
        )
        return pr.html_url
