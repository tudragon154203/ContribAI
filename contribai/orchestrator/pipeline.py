"""Main pipeline orchestrator.

Coordinates the full contribution flow:
discover → analyze → generate → PR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from contribai.analysis.analyzer import CodeAnalyzer
from contribai.core.config import ContribAIConfig
from contribai.core.models import (
    AnalysisResult,
    DiscoveryCriteria,
    PRResult,
    Repository,
)
from contribai.generator.engine import ContributionGenerator
from contribai.github.client import GitHubClient
from contribai.github.discovery import RepoDiscovery
from contribai.llm.provider import create_llm_provider
from contribai.orchestrator.memory import Memory
from contribai.pr.manager import PRManager

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a pipeline run."""

    repos_analyzed: int = 0
    findings_total: int = 0
    contributions_generated: int = 0
    prs_created: int = 0
    prs: list[PRResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class ContribPipeline:
    """Main orchestrator for the contribution pipeline."""

    def __init__(self, config: ContribAIConfig):
        self.config = config
        self._github: GitHubClient | None = None
        self._llm = None
        self._memory: Memory | None = None
        self._analyzer: CodeAnalyzer | None = None
        self._generator: ContributionGenerator | None = None
        self._pr_manager: PRManager | None = None
        self._discovery: RepoDiscovery | None = None

    async def _init_components(self):
        """Initialize all pipeline components."""
        # LLM
        self._llm = create_llm_provider(self.config.llm)

        # GitHub
        self._github = GitHubClient(
            token=self.config.github.token,
            rate_limit_buffer=self.config.github.rate_limit_buffer,
        )

        # Memory
        self._memory = Memory(self.config.storage.resolved_db_path)
        await self._memory.init()

        # Analyzer
        self._analyzer = CodeAnalyzer(
            llm=self._llm,
            github=self._github,
            config=self.config.analysis,
        )

        # Generator
        self._generator = ContributionGenerator(
            llm=self._llm,
            config=self.config.contribution,
        )

        # PR Manager
        self._pr_manager = PRManager(github=self._github)

        # Discovery
        self._discovery = RepoDiscovery(
            client=self._github,
            config=self.config.discovery,
        )

    async def _cleanup(self):
        """Clean up resources."""
        if self._github:
            await self._github.close()
        if self._llm:
            await self._llm.close()
        if self._memory:
            await self._memory.close()

    # ── Public API ─────────────────────────────────────────────────────────

    async def run(
        self,
        criteria: DiscoveryCriteria | None = None,
        dry_run: bool = False,
    ) -> PipelineResult:
        """Run the full pipeline: discover → analyze → generate → PR.

        Args:
            criteria: Optional custom discovery criteria
            dry_run: If True, analyze and generate but don't create PRs
        """
        await self._init_components()
        result = PipelineResult()
        run_id = await self._memory.start_run()

        try:
            # Check daily PR limit
            today_prs = await self._memory.get_today_pr_count()
            remaining_prs = self.config.github.max_prs_per_day - today_prs
            if remaining_prs <= 0 and not dry_run:
                logger.warning("Daily PR limit reached (%d)", self.config.github.max_prs_per_day)
                return result

            # 1. Discover repos
            logger.info("🔍 Discovering repositories...")
            repos = await self._discovery.discover(criteria)
            if not repos:
                logger.warning("No repositories found matching criteria")
                return result

            logger.info("Found %d candidate repositories", len(repos))

            # Limit to max repos per run
            repos = repos[: self.config.github.max_repos_per_run]

            # 2. Process each repo
            for repo in repos:
                # Skip if already analyzed recently
                if await self._memory.has_analyzed(repo.full_name):
                    logger.info("Skipping %s (already analyzed)", repo.full_name)
                    continue

                try:
                    repo_result = await self._process_repo(repo, dry_run, remaining_prs)
                    result.repos_analyzed += 1
                    result.findings_total += repo_result.findings_total
                    result.contributions_generated += repo_result.contributions_generated
                    result.prs_created += repo_result.prs_created
                    result.prs.extend(repo_result.prs)
                    remaining_prs -= repo_result.prs_created
                except Exception as e:
                    error = f"Error processing {repo.full_name}: {e}"
                    logger.error(error)
                    result.errors.append(error)

            # Log run
            await self._memory.finish_run(
                run_id,
                repos_analyzed=result.repos_analyzed,
                prs_created=result.prs_created,
                findings=result.findings_total,
                errors=len(result.errors),
            )

        finally:
            await self._cleanup()

        return result

    async def run_single(
        self,
        repo_url: str,
        dry_run: bool = False,
    ) -> PipelineResult:
        """Run the pipeline on a single specific repo.

        Args:
            repo_url: GitHub repository URL (e.g., https://github.com/owner/repo)
            dry_run: If True, analyze and generate but don't create PRs
        """
        # Parse URL
        parts = repo_url.rstrip("/").split("/")
        owner, name = parts[-2], parts[-1]

        await self._init_components()
        result = PipelineResult()

        try:
            repo = await self._github.get_repo_details(owner, name)
            repo_result = await self._process_repo(repo, dry_run)
            result.repos_analyzed = 1
            result.findings_total = repo_result.findings_total
            result.contributions_generated = repo_result.contributions_generated
            result.prs_created = repo_result.prs_created
            result.prs = repo_result.prs
        except Exception as e:
            result.errors.append(str(e))
            logger.error("Failed: %s", e)
        finally:
            await self._cleanup()

        return result

    async def analyze_only(self, repo_url: str) -> AnalysisResult | None:
        """Analyze a repo without generating contributions or PRs."""
        parts = repo_url.rstrip("/").split("/")
        owner, name = parts[-2], parts[-1]

        await self._init_components()
        try:
            repo = await self._github.get_repo_details(owner, name)
            return await self._analyzer.analyze(repo)
        finally:
            await self._cleanup()

    # ── Internal ───────────────────────────────────────────────────────────

    async def _process_repo(
        self, repo: Repository, dry_run: bool, max_prs: int = 5
    ) -> PipelineResult:
        """Process a single repository through the full pipeline."""
        result = PipelineResult()
        logger.info("=" * 60)
        logger.info("📦 Processing: %s", repo.full_name)

        # Analyze
        logger.info("🔬 Analyzing code...")
        analysis = await self._analyzer.analyze(repo)
        result.findings_total = len(analysis.findings)

        await self._memory.record_analysis(
            repo.full_name,
            repo.language or "unknown",
            repo.stars,
            len(analysis.findings),
        )

        if not analysis.findings:
            logger.info("No findings for %s", repo.full_name)
            return result

        logger.info(
            "Found %d issues (analyzed %d files in %.1fs)",
            len(analysis.findings),
            analysis.analyzed_files,
            analysis.analysis_duration_sec,
        )

        # Build context for generation
        file_tree = await self._github.get_file_tree(repo.owner, repo.name)
        relevant_files: dict[str, str] = {}
        for finding in analysis.top_findings[:5]:
            if finding.file_path and finding.file_path not in relevant_files:
                try:
                    content = await self._github.get_file_content(
                        repo.owner, repo.name, finding.file_path
                    )
                    relevant_files[finding.file_path] = content
                except Exception:
                    pass

        from contribai.core.models import RepoContext

        context = RepoContext(
            repo=repo,
            file_tree=file_tree,
            relevant_files=relevant_files,
        )

        # Generate contributions for top findings
        for finding in analysis.top_findings[:max_prs]:
            logger.info("🛠️ Generating fix for: %s", finding.title)
            contribution = await self._generator.generate(finding, context)

            if not contribution:
                continue

            result.contributions_generated += 1

            if dry_run:
                logger.info("🏃 [DRY RUN] Would create PR: %s", contribution.title)
                continue

            # Create PR
            try:
                logger.info("📤 Creating PR...")
                pr_result = await self._pr_manager.create_pr(contribution, repo)
                result.prs_created += 1
                result.prs.append(pr_result)

                # Record in memory
                await self._memory.record_pr(
                    repo=repo.full_name,
                    pr_number=pr_result.pr_number,
                    pr_url=pr_result.pr_url,
                    title=contribution.title,
                    pr_type=contribution.contribution_type.value,
                    branch=pr_result.branch_name,
                    fork=pr_result.fork_full_name,
                )
            except Exception as e:
                error = f"PR creation failed for {finding.title}: {e}"
                logger.error(error)
                result.errors.append(error)

        result.repos_analyzed = 1
        return result
