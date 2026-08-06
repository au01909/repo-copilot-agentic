"""Plan-aware file boosting: turns planner.Plan.sources into a score bonus for
matching files, so retrieval prefers e.g. README for repository-summary
questions without hard-filtering out everything else."""
import fnmatch
import posixpath

from . import config
from .planner import SOURCE_FILE_PATTERNS


def source_boost_score(file: str, plan) -> float:
    """Boost for a file path given a retrieval plan. 0.0 if no source in the
    plan has a matching glob (planner.SOURCE_FILE_PATTERNS) for this file."""
    if plan is None:
        return 0.0
    lowered = file.lower()
    basename = posixpath.basename(lowered)
    best = 0.0
    for source in plan.sources:
        patterns = SOURCE_FILE_PATTERNS.get(source)
        if not patterns:
            continue
        if any(fnmatch.fnmatch(lowered, pat) or fnmatch.fnmatch(basename, pat) for pat in patterns):
            weight = plan.source_weights.get(source) or config.SOURCE_BOOST_WEIGHTS.get(source, 0.0)
            best = max(best, weight)
    return best


def boosted_files(files, plan) -> list:
    """Files (in given order) that receive a nonzero boost under this plan."""
    if plan is None:
        return []
    return [f for f in files if source_boost_score(f, plan) > 0.0]
