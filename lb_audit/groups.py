"""Groups of rollouts, the two verifiers, and the advantages GRPO builds from them.

A rollout ends in one of three states:

* `GENUINE` — it solves the task and passes the cheap verifier;
* `HACK` — it passes the cheap verifier without solving the task;
* `FAIL` — it does not pass the cheap verifier.

The cheap verifier cannot tell the first two apart; that is what makes it cheap. The strong
verifier can, and is rationed. Everything in this package is about where to spend it.

Note what an audit costs. A rollout that already failed the cheap verifier needs no strong check —
there is nothing to disbelieve — so auditing a group costs one strong label per *passing* rollout.
Groups where little passed are therefore both the cheapest to audit and, as `advantage_mass` shows,
the ones where a hack collects the most advantage.
"""

import numpy as np

__all__ = ["FAIL", "GENUINE", "HACK", "Group", "advantages", "advantage_mass", "sample_groups"]

FAIL, GENUINE, HACK = 0, 1, 2


class Group:
    """One prompt's rollouts: what each one is, and which of them have been audited."""

    def __init__(self, kinds):
        self.kinds = np.asarray(kinds, dtype=np.int64)
        self.audited = np.zeros(self.kinds.shape, dtype=bool)

    def __len__(self):
        return int(self.kinds.size)

    @property
    def passes(self):
        """The rollouts the cheap verifier accepts — genuine and hacked alike."""
        return (self.kinds == GENUINE) | (self.kinds == HACK)

    @property
    def pass_count(self):
        return int(self.passes.sum())

    @property
    def hacks(self):
        return self.kinds == HACK

    @property
    def audit_cost(self):
        """Strong labels needed to audit this group: one per passing rollout."""
        return self.pass_count

    def rewards(self, correction=-1.0, pass_reward=1.0, fail_reward=0.0):
        """Rewards after whatever auditing has happened.

        An audited hack is *caught* and receives `correction`. An unaudited hack keeps the pass
        reward, because nothing has contradicted it.
        """
        values = np.where(self.passes, pass_reward, fail_reward).astype(np.float64)
        caught = self.hacks & self.audited
        values[caught] = correction
        return values


def advantages(rewards, normalise=False, eps=1e-8):
    """GRPO's group-relative advantage: centred, and optionally scaled by the group's spread."""
    rewards = np.asarray(rewards, dtype=np.float64)
    centred = rewards - rewards.mean()
    if not normalise:
        return centred
    return centred / (rewards.std() + eps)


def advantage_mass(group, correction=-1.0, normalise=False, **reward_kwargs):
    """The net advantage sitting on this group's hacks — the pull towards hacking.

    Positive means the update rewards hacking here. This is the quantity every audit policy in
    `policies` is trying to drive down, and the budget is the only reason not to drive it to zero
    everywhere at once.
    """
    values = advantages(
        group.rewards(correction=correction, **reward_kwargs), normalise=normalise
    )
    return float(values[group.hacks].sum())


def hack_advantage_if_unaudited(pass_count, size, pass_reward=1.0, fail_reward=0.0):
    """What one unaudited hack collects in a group with `pass_count` passes.

    `(pass_reward - fail_reward) * (1 - pass_count / size)`: strictly decreasing in the number of
    passes. A lone pass among failures takes nearly the whole group's advantage; one pass among
    many takes almost none. That is the sense in which hacks pay most where least passed.
    """
    spread = pass_reward - fail_reward
    return spread * (1.0 - pass_count / size)


def sample_groups(count, size, hack_rate, solve_rate, rng):
    """`count` groups of `size` rollouts drawn independently.

    Each rollout solves the task with probability `solve_rate`; failing that it hacks with
    probability `hack_rate`, and otherwise simply fails.
    """
    draws = rng.random((count, size))
    hack_draws = rng.random((count, size))
    kinds = np.full((count, size), FAIL, dtype=np.int64)
    solved = draws < solve_rate
    kinds[solved] = GENUINE
    hacked = (~solved) & (hack_draws < hack_rate)
    kinds[hacked] = HACK
    return [Group(row) for row in kinds]
