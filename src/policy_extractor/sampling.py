"""Stratified spot-check sampling of auto-approved records (CLAUDE.md §7).

Auto-approval is only trustworthy if someone keeps checking it. Sampling per policy type (rather
than over the whole run) guarantees that a rare type is never skipped, which is exactly where a
layout- or product-specific failure would otherwise hide.
"""

import random
from collections import defaultdict
from collections.abc import Mapping

UNKNOWN_TYPE = "<unknown>"


def select_spot_checks(type_by_doc: Mapping[str, str | None], share: float, seed: int) -> set[str]:
    """Pick documents to spot-check from `{doc_id: policy_type}` of auto-approved records.

    Each policy type contributes `round(share * n)` documents and never fewer than one; a share
    of 0 turns sampling off. Deterministic for a given seed, independent of input order.
    """
    if share <= 0:
        return set()
    strata: dict[str, list[str]] = defaultdict(list)
    for doc_id, policy_type in type_by_doc.items():
        strata[policy_type or UNKNOWN_TYPE].append(doc_id)
    selected: set[str] = set()
    for policy_type in sorted(strata):
        doc_ids = sorted(strata[policy_type])
        count = min(len(doc_ids), max(1, round(share * len(doc_ids))))
        selected.update(random.Random(f"{seed}:{policy_type}").sample(doc_ids, count))
    return selected
