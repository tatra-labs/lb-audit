"""Tests for the audit policies, the corrections, and the algebra they rest on.

The claims being checked are the ones the method stands or falls on: that an unaudited hack is
worth more where less passed, that low-pass ordering is both cheaper and more effective than any
other group ordering, and that reflecting a caught hack's reward reverses the pull rather than
merely cancelling it.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lb_audit import (  # noqa: E402
    FAIL,
    GENUINE,
    HACK,
    Group,
    advantage_mass,
    advantages,
    audit,
    caught_reward,
    hack_advantage_if_unaudited,
    sample_groups,
)
from lb_audit.policies import POLICIES  # noqa: E402


def group_of(genuine=0, hacks=0, fails=0):
    return Group([GENUINE] * genuine + [HACK] * hacks + [FAIL] * fails)


# --- the algebra the method rests on ---------------------------------------------------------


def test_advantages_are_centred():
    values = advantages(np.array([1.0, 0.0, 0.0, 1.0]))
    assert float(values.sum()) == pytest.approx(0.0, abs=1e-12)


def test_a_hack_is_worth_more_where_less_passed():
    """The whole reason to audit low-pass groups first."""
    worth = [hack_advantage_if_unaudited(count, 8) for count in range(1, 9)]
    assert worth == sorted(worth, reverse=True)
    assert worth[0] == pytest.approx(0.875)
    assert worth[-1] == pytest.approx(0.0)


def test_the_closed_form_matches_the_real_advantage():
    for passes in range(1, 9):
        group = group_of(hacks=1, genuine=passes - 1, fails=8 - passes)
        values = advantages(group.rewards())
        got = float(values[group.hacks][0])
        assert got == pytest.approx(hack_advantage_if_unaudited(passes, 8))


def test_auditing_only_costs_labels_for_passing_rollouts():
    group = group_of(genuine=2, hacks=1, fails=5)
    assert group.audit_cost == 3
    assert group_of(fails=8).audit_cost == 0


def test_an_unaudited_hack_keeps_the_pass_reward():
    group = group_of(hacks=1, fails=7)
    assert float(group.rewards()[0]) == 1.0
    group.audited |= group.passes
    assert float(group.rewards(correction=-1.0)[0]) == -1.0


# --- the corrections -------------------------------------------------------------------------


def test_reflect_reverses_the_pull_and_fail_only_cancels_it():
    group = group_of(hacks=2, fails=6)
    group.audited |= group.passes
    assert advantage_mass(group, correction=caught_reward("fail")) == pytest.approx(0.0)
    assert advantage_mass(group, correction=caught_reward("reflect")) < -0.1


def test_a_group_that_is_all_hacks_gives_no_signal_either_way():
    """Everything in it receives the same reward, so centring leaves nothing."""
    group = group_of(hacks=8)
    group.audited |= group.passes
    for name in ("fail", "reflect"):
        assert advantage_mass(group, correction=caught_reward(name)) == pytest.approx(0.0)
    assert advantage_mass(group, correction=1.0) == pytest.approx(0.0)


def test_the_unbiased_correction_is_unbiased():
    """A hack's expected reward under random auditing must equal the failure reward."""
    for probability in (1.0, 0.5, 0.25, 0.1):
        caught = caught_reward("unbiased", audit_probability=probability)
        expected = probability * caught + (1.0 - probability) * 1.0
        assert expected == pytest.approx(0.0, abs=1e-9)


def test_the_unbiased_correction_grows_as_the_budget_shrinks():
    values = [caught_reward("unbiased", audit_probability=q) for q in (1.0, 0.5, 0.2, 0.1)]
    assert values == [pytest.approx(v) for v in sorted(values, reverse=True)]
    assert caught_reward("unbiased", audit_probability=0.1) == pytest.approx(-9.0)


def test_unknown_names_are_refused():
    with pytest.raises(ValueError):
        caught_reward("invent-a-number")
    with pytest.raises(ValueError):
        audit([], "audit-everything-twice", 10)


# --- the audit policies ----------------------------------------------------------------------


def test_low_pass_takes_the_cheapest_groups_first():
    groups = [group_of(genuine=n, fails=8 - n) for n in (6, 1, 4, 2)]
    spent = audit(groups, "low_pass", budget=3)
    assert [g.audited.any() for g in groups] == [False, True, False, True]
    assert spent["labels"] == 3 and spent["groups"] == 2


def test_high_pass_spends_the_whole_budget_on_one_expensive_group():
    """Where low-pass covers several groups, high-pass buys one, and it is the one where a
    hack was worth least anyway."""
    costs = (6, 1, 4, 2)
    dear = [group_of(hacks=n, fails=8 - n) for n in costs]
    cheap = [group_of(hacks=n, fails=8 - n) for n in costs]
    expensive = audit(dear, "high_pass", budget=6)
    frugal = audit(cheap, "low_pass", budget=6)
    assert expensive["groups"] == 1 and expensive["labels"] == 6
    assert frugal["groups"] == 2 and frugal["labels"] == 3
    # And the cheap ones were the groups whose hacks carried the most advantage.
    assert hack_advantage_if_unaudited(1, 8) > hack_advantage_if_unaudited(6, 8)


def test_a_budget_that_fits_nothing_audits_nothing():
    groups = [group_of(genuine=4, fails=4) for _ in range(3)]
    spent = audit(groups, "low_pass", budget=2)
    assert spent == {"labels": 0, "groups": 0}
    assert not any(g.audited.any() for g in groups)


def test_no_policy_ever_audits_a_failing_rollout():
    rng = np.random.default_rng(0)
    for name in POLICIES:
        groups = sample_groups(40, 8, 0.4, 0.35, np.random.default_rng(1))
        audit(groups, name, budget=50, rng=rng)
        for group in groups:
            assert not bool((group.audited & ~group.passes).any())


def test_no_policy_spends_more_than_its_budget():
    rng = np.random.default_rng(0)
    for name in ("low_pass", "random", "high_pass"):
        for budget in (0, 5, 17, 40):
            groups = sample_groups(30, 8, 0.4, 0.35, np.random.default_rng(2))
            spent = audit(groups, name, budget=budget, rng=rng)
            assert spent["labels"] <= budget


def test_full_audits_everything_and_none_audits_nothing():
    groups = sample_groups(10, 8, 0.4, 0.35, np.random.default_rng(3))
    audit(groups, "full", budget=0)
    assert all(bool((g.audited == g.passes).all()) for g in groups)

    groups = sample_groups(10, 8, 0.4, 0.35, np.random.default_rng(3))
    audit(groups, "none", budget=1000)
    assert not any(g.audited.any() for g in groups)


# --- the claim the paper is about --------------------------------------------------------------


@pytest.mark.parametrize("correction", ["fail", "reflect"])
@pytest.mark.parametrize("budget_fraction", [0.1, 0.2, 0.3, 0.5])
def test_low_pass_leaves_less_advantage_on_hacks_than_any_other_ordering(
    correction, budget_fraction
):
    """The paper's claim, at a fixed policy: audit where hacks pay, and pay less for it."""
    results = {}
    for name in ("low_pass", "random", "high_pass"):
        masses = []
        for seed in range(12):
            rng = np.random.default_rng(seed)
            groups = sample_groups(128, 8, 0.4, 0.35, rng)
            total = sum(g.audit_cost for g in groups)
            audit(groups, name, int(round(budget_fraction * total)), rng)
            audited = sum(int(g.audited.sum()) for g in groups)
            reward = caught_reward(correction, audit_probability=audited / total)
            masses.append(
                sum(advantage_mass(g, correction=reward) for g in groups) / len(groups)
            )
        results[name] = float(np.mean(masses))
    assert results["low_pass"] < results["random"] < results["high_pass"]


def test_low_pass_audits_more_groups_for_the_same_labels():
    """The other half of the bargain: low-pass groups are the cheap ones."""
    rng = np.random.default_rng(0)
    counts = {}
    for name in ("low_pass", "random", "high_pass"):
        groups = sample_groups(128, 8, 0.4, 0.35, np.random.default_rng(4))
        total = sum(g.audit_cost for g in groups)
        counts[name] = audit(groups, name, int(0.25 * total), rng)["groups"]
    assert counts["low_pass"] > counts["random"] > counts["high_pass"]


def test_training_without_audits_learns_to_hack():
    from lb_audit import train

    run = train(policy="none", correction="fail", steps=200, seed=0)
    assert run.final["hack_probability"] > 0.9
    assert run.final["held_out_accuracy"] < 0.05


def test_a_full_audit_with_reflection_stops_it():
    from lb_audit import train

    run = train(policy="full", correction="reflect", steps=200, solve_rate=0.35, seed=0)
    assert run.final["hack_probability"] < 0.05
    assert run.final["held_out_accuracy"] > 0.3
