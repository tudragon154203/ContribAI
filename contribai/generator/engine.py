"""LLM-powered contribution generator.

Takes findings from the analysis engine and generates
actual code changes, tests, and commit messages that
follow the target repository's coding conventions.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime

from contribai.core.config import ContributionConfig
from contribai.core.models import (
    Contribution,
    ContributionType,
    FileChange,
    Finding,
    RepoContext,
)
from contribai.llm.context import build_repo_context_prompt
from contribai.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class ContributionGenerator:
    """Generate code contributions from analysis findings."""

    def __init__(self, llm: LLMProvider, config: ContributionConfig):
        self._llm = llm
        self._config = config

    async def generate(self, finding: Finding, context: RepoContext) -> Contribution | None:
        """Generate a contribution for a single finding.

        Steps:
        1. Build context-aware prompt
        2. Get LLM to generate the fix
        3. Parse structured output into FileChanges
        4. Generate commit message
        5. Self-review the generated code
        """
        try:
            # 1 & 2: Generate the fix
            prompt = self._build_generation_prompt(finding, context)
            system = self._build_system_prompt(context)

            response = await self._llm.complete(prompt, system=system, temperature=0.2)

            # 3: Parse output
            changes = self._parse_changes(response)
            if not changes:
                logger.warning("No valid changes parsed for finding: %s", finding.title)
                return None

            # 4: Generate commit message
            commit_msg = await self._generate_commit_message(finding, changes, context)

            # 5: Generate branch name
            branch_name = self._generate_branch_name(finding)

            # Build the contribution
            contribution = Contribution(
                finding=finding,
                contribution_type=finding.type,
                title=self._generate_pr_title(finding),
                description=finding.description,
                changes=changes,
                commit_message=commit_msg,
                branch_name=branch_name,
                generated_at=datetime.utcnow(),
            )

            # 6: Self-review
            review_passed = await self._self_review(contribution, context)
            if not review_passed:
                logger.warning("Self-review failed for: %s", finding.title)
                return None

            logger.info(
                "Generated contribution: %s (%d files changed)",
                contribution.title,
                contribution.total_files_changed,
            )
            return contribution

        except Exception as e:
            logger.error("Failed to generate contribution for %s: %s", finding.title, e)
            return None

    def _build_system_prompt(self, context: RepoContext) -> str:
        """Build system prompt with repository context."""
        repo_context = build_repo_context_prompt(context, max_tokens=4000)
        return (
            "You are an expert open-source contributor. You generate high-quality, "
            "production-ready code changes that follow the target repository's coding "
            "conventions and best practices.\n\n"
            "IMPORTANT RULES:\n"
            "- Match the existing code style exactly (indentation, naming, patterns)\n"
            "- Make minimal, focused changes that fix exactly one issue\n"
            "- Include proper error handling\n"
            "- Do NOT break existing functionality\n"
            "- Do NOT add unnecessary dependencies\n"
            "- Write clear, self-documenting code\n\n"
            f"REPOSITORY CONTEXT:\n{repo_context}"
        )

    def _build_generation_prompt(self, finding: Finding, context: RepoContext) -> str:
        """Build the generation prompt based on finding type."""
        # Get the current file content if available
        current_content = context.relevant_files.get(finding.file_path, "")

        type_instructions = {
            ContributionType.SECURITY_FIX: (
                "Fix this SECURITY vulnerability. Ensure the fix is complete "
                "and doesn't introduce new vulnerabilities."
            ),
            ContributionType.CODE_QUALITY: (
                "Improve the CODE QUALITY. Make the code cleaner, more maintainable, "
                "and more robust. Keep changes minimal and focused."
            ),
            ContributionType.DOCS_IMPROVE: (
                "Improve the DOCUMENTATION. Add missing docstrings, improve README sections, "
                "or fix documentation issues. Be thorough but concise."
            ),
            ContributionType.UI_UX_FIX: (
                "Fix this UI/UX issue. Improve accessibility, user experience, or visual design. "
                "Follow WCAG guidelines where applicable."
            ),
            ContributionType.PERFORMANCE_OPT: (
                "Optimize PERFORMANCE. Reduce time/space complexity, "
                "eliminate wasteful operations, or improve resource usage."
            ),
            ContributionType.FEATURE_ADD: (
                "Add this FEATURE. Keep the implementation clean, well-structured, and consistent "
                "with the existing codebase patterns."
            ),
            ContributionType.REFACTOR: (
                "REFACTOR this code. Improve structure and readability without changing behavior."
            ),
        }

        instruction = type_instructions.get(finding.type, "Fix this issue.")

        prompt = (
            f"## Task\n{instruction}\n\n"
            f"## Finding\n"
            f"- **Title**: {finding.title}\n"
            f"- **Severity**: {finding.severity.value}\n"
            f"- **File**: {finding.file_path}\n"
            f"- **Description**: {finding.description}\n"
        )

        if finding.suggestion:
            prompt += f"- **Suggestion**: {finding.suggestion}\n"

        if current_content:
            prompt += (
                f"\n## Current File Content ({finding.file_path})\n"
                f"```\n{current_content[:4000]}\n```\n"
            )

        prompt += (
            "\n## Output Format\n"
            "Return your changes as a JSON object with this structure:\n"
            "```json\n"
            "{\n"
            '  "changes": [\n'
            "    {\n"
            '      "path": "path/to/file.py",\n'
            '      "content": "full new content of the file",\n'
            '      "is_new_file": false\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n"
            "Include ONLY the files that need changes. Provide the FULL content of each "
            "changed file, not just the diff.\n"
        )

        return prompt

    def _parse_changes(self, response: str) -> list[FileChange]:
        """Parse LLM response into FileChange objects."""
        changes: list[FileChange] = []

        try:
            # Try to extract JSON from the response
            json_match = re.search(r"```json\s*\n(.*?)\n\s*```", response, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r"\{[\s\S]*\"changes\"[\s\S]*\}", response)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    return []

            data = json.loads(json_text)
            raw_changes = data.get("changes", [])

            for item in raw_changes:
                if not isinstance(item, dict) or "path" not in item or "content" not in item:
                    continue

                changes.append(
                    FileChange(
                        path=item["path"],
                        new_content=item["content"],
                        is_new_file=item.get("is_new_file", False),
                    )
                )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to parse changes JSON: %s", e)

        # Enforce max files limit
        if len(changes) > self._config.max_files_per_pr:
            logger.warning(
                "Too many files changed (%d > %d), truncating",
                len(changes),
                self._config.max_files_per_pr,
            )
            changes = changes[: self._config.max_files_per_pr]

        return changes

    async def _generate_commit_message(
        self, finding: Finding, changes: list[FileChange], context: RepoContext
    ) -> str:
        """Generate a conventional commit message."""
        type_prefixes = {
            ContributionType.SECURITY_FIX: "fix(security)",
            ContributionType.CODE_QUALITY: "refactor",
            ContributionType.DOCS_IMPROVE: "docs",
            ContributionType.UI_UX_FIX: "fix(ui)",
            ContributionType.PERFORMANCE_OPT: "perf",
            ContributionType.FEATURE_ADD: "feat",
            ContributionType.REFACTOR: "refactor",
        }

        prefix = type_prefixes.get(finding.type, "fix")
        files = ", ".join(c.path.split("/")[-1] for c in changes[:3])

        if self._config.commit_convention == "conventional":
            return (
                f"{prefix}: {finding.title.lower()}\n\n"
                f"{finding.description}\n\n"
                f"Affected files: {files}"
            )
        elif self._config.commit_convention == "angular":
            scope = changes[0].path.split("/")[0] if changes else ""
            return f"{prefix}({scope}): {finding.title.lower()}"
        else:
            return finding.title

    def _generate_branch_name(self, finding: Finding) -> str:
        """Generate a clean branch name from finding."""
        prefix_map = {
            ContributionType.SECURITY_FIX: "fix/security",
            ContributionType.CODE_QUALITY: "improve/quality",
            ContributionType.DOCS_IMPROVE: "docs",
            ContributionType.UI_UX_FIX: "fix/ui",
            ContributionType.PERFORMANCE_OPT: "perf",
            ContributionType.FEATURE_ADD: "feat",
            ContributionType.REFACTOR: "refactor",
        }
        prefix = prefix_map.get(finding.type, "fix")
        # Clean title for branch name
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", finding.title.lower()).strip("-")[:40]
        return f"contribai/{prefix}/{slug}"

    def _generate_pr_title(self, finding: Finding) -> str:
        """Generate a clear PR title."""
        type_labels = {
            ContributionType.SECURITY_FIX: "🔒 Security",
            ContributionType.CODE_QUALITY: "✨ Quality",
            ContributionType.DOCS_IMPROVE: "📝 Docs",
            ContributionType.UI_UX_FIX: "🎨 UI/UX",
            ContributionType.PERFORMANCE_OPT: "⚡ Performance",
            ContributionType.FEATURE_ADD: "🚀 Feature",
            ContributionType.REFACTOR: "♻️ Refactor",
        }
        label = type_labels.get(finding.type, "🔧 Fix")
        return f"{label}: {finding.title}"

    async def _self_review(self, contribution: Contribution, context: RepoContext) -> bool:
        """Have the LLM self-review the generated contribution."""
        changes_summary = "\n".join(
            f"- {c.path} ({'new' if c.is_new_file else 'modified'})" for c in contribution.changes
        )

        prompt = (
            "Review the following code contribution for quality:\n\n"
            f"**Title**: {contribution.title}\n"
            f"**Type**: {contribution.contribution_type.value}\n"
            f"**Changes**:\n{changes_summary}\n\n"
            "For each changed file:\n"
        )
        for change in contribution.changes[:5]:
            prompt += f"\n### {change.path}\n```\n{change.new_content[:2000]}\n```\n"

        prompt += (
            "\nAnswer these questions:\n"
            "1. Does the change correctly fix the described issue?\n"
            "2. Does it introduce any new bugs or security issues?\n"
            "3. Does it follow good coding practices?\n"
            "4. Is the change minimal and focused?\n\n"
            "Reply with APPROVE or REJECT followed by brief reasoning."
        )

        try:
            response = await self._llm.complete(prompt, temperature=0.1)
            approved = "APPROVE" in response.upper()
            if not approved:
                logger.info("Self-review rejected: %s", response[:200])
            return approved
        except Exception as e:
            logger.warning("Self-review failed, approving by default: %s", e)
            return True  # Don't block on review failures
