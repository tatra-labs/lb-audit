"""Bias and variance of the update, measured at a policy that is not moving.

The paper's closing point is that an audit scheme is not described by its expected update alone:
two schemes can both point away from hacking and differ in how noisy the estimate is. During
training that is hard to see, because a policy in the middle of changing its mind produces large
updates for reasons that have nothing to do with the estimator.

So hold the policy still. Draw many independent batches at a fixed hack rate, form the update each
time, and report its mean and its spread. The mean is what the scheme is aiming at; the spread is
what each individual step actually does.

    python experiments/variance.py
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lb_audit import advantages, audit, caught_reward, sample_groups  # noqa: E402

ARMS = [
    ("random", "fail"),
    ("low_pass", "fail"),
    ("random", "reflect"),
    ("low_pass", "reflect"),
    ("random", "unbiased"),
    ("low_pass", "unbiased"),
]


def one_batch(policy, correction, fraction, args, seed):
    """The hack advantage this batch contributes, under one scheme."""
    rng = np.random.default_rng(seed)
    groups = sample_groups(args.groups, args.group_size, args.hack_rate, args.solve_rate, rng)
    total_cost = sum(group.audit_cost for group in groups)
    budget = int(round(fraction * total_cost))
    audit(groups, policy, budget, rng)

    audited = sum(int(group.audited.sum()) for group in groups)
    probability = audited / total_cost if total_cost else 0.0
    reward = caught_reward(correction, audit_probability=probability)

    mass = 0.0
    for group in groups:
        values = advantages(group.rewards(correction=reward))
        mass += float(values[group.hacks].sum())
    return mass / len(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--groups", type=int, default=16, help="groups per batch, as in training")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--hack-rate", type=float, default=0.4)
    parser.add_argument("--solve-rate", type=float, default=0.35)
    parser.add_argument("--budget", type=float, default=0.3)
    parser.add_argument("--batches", type=int, default=4000)
    parser.add_argument("--out", default="results/variance.json")
    args = parser.parse_args()

    print("hack advantage per group at a fixed policy, %d batches of %d groups, budget %.0f%%\n"
          % (args.batches, args.groups, 100 * args.budget))
    print("%20s %10s %10s %12s %12s" % ("scheme", "mean", "sd", "sd / |mean|", "P(wrong sign)"))

    rows = []
    for policy, correction in ARMS:
        samples = np.array([
            one_batch(policy, correction, args.budget, args, seed)
            for seed in range(args.batches)
        ])
        mean, sd = float(samples.mean()), float(samples.std())
        # How often a single step pushes towards hacking although the scheme points away.
        wrong = float((samples > 0).mean()) if mean < 0 else float((samples < 0).mean())
        row = {
            "policy": policy,
            "correction": correction,
            "mean": mean,
            "sd": sd,
            "relative_sd": sd / abs(mean) if mean else float("inf"),
            "wrong_sign": wrong,
        }
        rows.append(row)
        print("%20s %10.4f %10.4f %12.2f %11.1f%%" % (
            "%s/%s" % (policy, correction), mean, sd, row["relative_sd"], 100 * wrong), flush=True)

    print("\n`mean` below zero means the scheme points away from hacking. `sd / |mean|` is how")
    print("noisy each individual step is relative to that signal, and `P(wrong sign)` is how")
    print("often one batch pushes the policy the wrong way regardless.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"args": vars(args), "rows": rows}, indent=2) + "\n")
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
