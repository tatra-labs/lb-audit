"""What reward a hack receives once the strong verifier has caught it.

The audit policy decides *where* to look. The correction decides what finding something is worth,
and the two are not independent: a correction only acts on the rollouts the budget reached, so the
smaller the budget, the more each catch has to carry.

Three choices, and the difference between them is the whole bias-variance argument:

* `fail` treats a caught hack as though it had simply failed. Honest, mild, and it only removes the
  hack's advantage rather than reversing it.
* `reflect` puts the caught hack as far below the failure reward as a pass is above it. A caught
  hack is then actively worse than not answering, so the update pushes away from hacking rather
  than merely declining to push towards it.
* `unbiased` scales the penalty by one over the audit probability, so that a hack's *expected*
  reward equals what it would have been under a full audit. That is unbiased by construction and,
  at small budgets, enormous: at a 10% audit rate a caught hack is worth -9, and every update that
  happens to contain one is dominated by it.
"""

__all__ = ["CORRECTIONS", "caught_reward"]


def _fail(pass_reward, fail_reward, audit_probability):
    return fail_reward


def _reflect(pass_reward, fail_reward, audit_probability):
    return 2.0 * fail_reward - pass_reward


def _unbiased(pass_reward, fail_reward, audit_probability, target=None):
    """The value that makes E[reward of a hack] equal `target` under random auditing.

    `q * caught + (1 - q) * pass = target`, so `caught = (target - (1 - q) * pass) / q`.
    """
    target = fail_reward if target is None else target
    probability = max(float(audit_probability), 1e-6)
    return (target - (1.0 - probability) * pass_reward) / probability


CORRECTIONS = {"fail": _fail, "reflect": _reflect, "unbiased": _unbiased}


def caught_reward(name, pass_reward=1.0, fail_reward=0.0, audit_probability=1.0):
    if name not in CORRECTIONS:
        raise ValueError("unknown correction: %r" % name)
    return CORRECTIONS[name](pass_reward, fail_reward, audit_probability)
