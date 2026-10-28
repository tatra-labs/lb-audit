"""Who to spend the strong verifier on, given a budget of strong labels.

Every policy takes the batch of groups and a label budget, and marks rollouts as audited. They all
audit whole groups — a half-audited group still has unchallenged passes in it — and they only ever
audit passing rollouts, since a rollout that already failed has nothing to check.

The only thing that separates them is the order they take groups in.
"""

import numpy as np

__all__ = ["POLICIES", "audit", "full", "high_pass", "low_pass", "none", "random_groups"]


def _spend(groups, order, budget):
    """Audit whole groups in `order` until the next one would not fit in the budget."""
    spent = 0
    audited = 0
    for index in order:
        group = groups[index]
        cost = group.audit_cost
        if cost == 0:
            continue
        if spent + cost > budget:
            continue
        group.audited |= group.passes
        spent += cost
        audited += 1
    return {"labels": spent, "groups": audited}


def low_pass(groups, budget, rng=None):
    """The paper's rule: fewest weak passes first.

    Both halves of the bargain point the same way. A hack in a group where little passed collects
    the largest advantage, and that group is also the cheapest to audit, because there are fewer
    passes to check.
    """
    costs = np.array([group.pass_count for group in groups])
    order = np.argsort(costs, kind="stable")
    return _spend(groups, order, budget)


def high_pass(groups, budget, rng=None):
    """The opposite order, kept because it is the natural thing to try and it is the worst."""
    costs = np.array([group.pass_count for group in groups])
    order = np.argsort(-costs, kind="stable")
    return _spend(groups, order, budget)


def random_groups(groups, budget, rng=None):
    """Whole groups in a random order — the baseline the paper's claim is against."""
    rng = np.random.default_rng(0) if rng is None else rng
    order = rng.permutation(len(groups))
    return _spend(groups, order, budget)


def full(groups, budget=None, rng=None):
    """Audit everything. The thing a budget exists to avoid."""
    spent = 0
    for group in groups:
        group.audited |= group.passes
        spent += group.audit_cost
    return {"labels": spent, "groups": len(groups)}


def none(groups, budget=None, rng=None):
    """Audit nothing: the cheap verifier alone."""
    return {"labels": 0, "groups": 0}


POLICIES = {
    "low_pass": low_pass,
    "random": random_groups,
    "high_pass": high_pass,
    "full": full,
    "none": none,
}


def audit(groups, policy, budget, rng=None):
    """Run one of `POLICIES` by name. Returns what it spent."""
    if policy not in POLICIES:
        raise ValueError("unknown audit policy: %r" % policy)
    return POLICIES[policy](groups, budget, rng)
