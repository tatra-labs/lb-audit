"""A GRPO loop small enough to see through.

There is no language model here. The policy is three logits over three things a rollout can do —
attempt the task honestly, hack the cheap verifier, or fail — and the learning rule is the one GRPO
uses: sample a group, centre the rewards within it, and push each action by its advantage.

That is enough to reproduce the behaviour the audit policies exist to prevent. With no auditing,
hacking always collects the pass reward, so its advantage is positive whenever anything failed, and
the policy walks to a hack rate of one. Everything else in this package is about taking that
advantage away on a budget.

`held_out_accuracy` is what the policy is actually worth: how often it solves the task, which only
the honest action does.
"""

import numpy as np

from .corrections import caught_reward
from .groups import FAIL, GENUINE, HACK, Group, advantages
from .policies import audit

__all__ = ["Run", "train"]

ATTEMPT, HACK_ACTION, GIVE_UP = 0, 1, 2


def _softmax(logits):
    shifted = logits - logits.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


class Run:
    """What one training run leaves behind."""

    def __init__(self):
        self.history = []

    def record(self, **fields):
        self.history.append(fields)

    def column(self, name):
        return np.array([row[name] for row in self.history])

    @property
    def final(self):
        return self.history[-1]


def _outcomes(actions, solve_rate, rng):
    kinds = np.full(actions.shape, FAIL, dtype=np.int64)
    attempting = actions == ATTEMPT
    kinds[attempting & (rng.random(actions.shape) < solve_rate)] = GENUINE
    kinds[actions == HACK_ACTION] = HACK
    return kinds


def train(
    policy="low_pass",
    correction="reflect",
    budget_fraction=0.25,
    steps=400,
    batch=16,
    group_size=8,
    solve_rate=0.35,
    learning_rate=0.5,
    normalise=False,
    initial_hack_logit=0.0,
    seed=0,
    record_every=5,
):
    """Train the toy policy under one audit policy and one correction.

    `budget_fraction` is a share of the strong labels a full audit of the batch would have used,
    so 0.25 means a quarter of the labels the `full` policy would spend.
    """
    rng = np.random.default_rng(seed)
    logits = np.array([0.0, float(initial_hack_logit), 0.0])
    run = Run()
    labels_used = 0

    for step in range(steps + 1):
        probabilities = _softmax(logits)
        actions = rng.choice(3, size=(batch, group_size), p=probabilities)
        kinds = _outcomes(actions, solve_rate, rng)
        groups = [Group(row) for row in kinds]

        total_cost = sum(group.audit_cost for group in groups)
        budget = int(round(budget_fraction * total_cost))
        spent = audit(groups, policy, budget, rng)
        labels_used += spent["labels"]

        audited_passes = sum(int(group.audited.sum()) for group in groups)
        audit_probability = audited_passes / total_cost if total_cost else 0.0
        reward_for_caught = caught_reward(
            correction, audit_probability=audit_probability
        )

        gradient = np.zeros(3)
        hack_mass = 0.0
        update_terms = []
        for group, row in zip(groups, actions):
            values = advantages(
                group.rewards(correction=reward_for_caught), normalise=normalise
            )
            hack_mass += float(values[group.hacks].sum())
            for advantage, action in zip(values, row):
                # d/dlogits of log softmax: one-hot minus the policy.
                term = -advantage * probabilities
                term[action] += advantage
                gradient += term
                update_terms.append(term)
        gradient /= batch * group_size

        if step % record_every == 0:
            run.record(
                step=step,
                hack_probability=float(probabilities[HACK_ACTION]),
                attempt_probability=float(probabilities[ATTEMPT]),
                held_out_accuracy=float(probabilities[ATTEMPT] * solve_rate),
                hack_advantage_mass=hack_mass / batch,
                audit_probability=audit_probability,
                labels_used=labels_used,
                update_norm=float(np.linalg.norm(gradient)),
                update_variance=float(np.var(np.array(update_terms), axis=0).sum()),
                caught_reward=float(reward_for_caught),
            )

        logits = logits + learning_rate * gradient
        logits -= logits.max()

    return run
