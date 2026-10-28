"""How much pull towards hacking each audit policy removes, per strong label spent.

No training here: draw a batch of groups, spend a budget on them, and measure what advantage is
left sitting on the hacks. That number is what the policy gradient will act on, so a policy that
cannot drive it to zero cannot stop hacking however long you train.

    python experiments/advantage.py
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lb_audit import advantage_mass, audit, caught_reward, sample_groups  # noqa: E402

POLICIES = ["none", "high_pass", "random", "low_pass", "full"]


def measure(policy, correction, fraction, args, seed):
    rng = np.random.default_rng(seed)
    groups = sample_groups(args.groups, args.group_size, args.hack_rate, args.solve_rate, rng)
    total_cost = sum(group.audit_cost for group in groups)
    budget = int(round(fraction * total_cost))
    spent = audit(groups, policy, budget, rng)

    audited = sum(int(group.audited.sum()) for group in groups)
    probability = audited / total_cost if total_cost else 0.0
    reward = caught_reward(correction, audit_probability=probability)

    mass = sum(advantage_mass(group, correction=reward) for group in groups)
    hacks = sum(int(group.hacks.sum()) for group in groups)
    caught = sum(int((group.hacks & group.audited).sum()) for group in groups)
    return {
        "policy": policy,
        "correction": correction,
        "budget_fraction": fraction,
        "labels": spent["labels"],
        "label_fraction": spent["labels"] / total_cost if total_cost else 0.0,
        "groups_audited": spent["groups"],
        "mass_per_group": mass / len(groups),
        "hacks": hacks,
        "hacks_caught": caught,
        "catch_rate": caught / hacks if hacks else 0.0,
    }


def crossing(policy, correction, args, lo=0.0, hi=1.0, rounds=18):
    """The smallest label budget at which this policy drives the hack advantage to zero."""
    def mass(fraction):
        return float(
            np.mean(
                [
                    measure(policy, correction, fraction, args, seed)["mass_per_group"]
                    for seed in range(args.seeds)
                ]
            )
        )

    if mass(hi) > 0:
        return None
    for _ in range(rounds):
        middle = 0.5 * (lo + hi)
        if mass(middle) > 0:
            lo = middle
        else:
            hi = middle
    spent = float(
        np.mean(
            [
                measure(policy, correction, hi, args, seed)["label_fraction"]
                for seed in range(args.seeds)
            ]
        )
    )
    return spent


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--groups", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--hack-rate", type=float, default=0.4)
    parser.add_argument("--solve-rate", type=float, default=0.35)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out", default="results/advantage.json")
    parser.add_argument("--figure", default="figures/advantage.png")
    args = parser.parse_args()

    fractions = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for correction in ("fail", "reflect"):
        print("\ncorrection: %s" % correction)
        print("%10s  %s" % ("budget", "  ".join("%12s" % p for p in POLICIES)))
        for fraction in fractions:
            line = {}
            for policy in POLICIES:
                trials = [
                    measure(policy, correction, fraction, args, seed)
                    for seed in range(args.seeds)
                ]
                row = {
                    key: float(np.mean([trial[key] for trial in trials]))
                    for key in ("mass_per_group", "label_fraction", "catch_rate")
                }
                row.update(policy=policy, correction=correction, budget_fraction=fraction)
                rows.append(row)
                line[policy] = row["mass_per_group"]
            print("%9.0f%%  %s" % (
                100 * fraction, "  ".join("%12.4f" % line[p] for p in POLICIES)), flush=True)

    print("\n(advantage mass left on hacks, per group; lower is better, negative actively")
    print("pushes away from hacking. `none` and `full` ignore the budget by definition.)")

    print("\nStrong labels needed to reach zero advantage on hacks:")
    print("%12s %14s %14s" % ("policy", "fail", "reflect"))
    crossings = {}
    for policy in ("low_pass", "random", "high_pass"):
        spent = {}
        for correction in ("fail", "reflect"):
            spent[correction] = crossing(policy, correction, args)
            crossings["%s/%s" % (policy, correction)] = spent[correction]
        print("%12s %13s %14s" % (
            policy,
            "%.1f%%" % (100 * spent["fail"]) if spent["fail"] is not None else "never",
            "%.1f%%" % (100 * spent["reflect"]) if spent["reflect"] is not None else "never",
        ), flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(
                {"args": vars(args), "rows": rows, "crossings": crossings}, indent=2
            ) + "\n")
        print("wrote %s" % args.out)
    if args.figure:
        plot(rows, args.figure)


def plot(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {
        "low_pass": "#2f6fc2",
        "random": "#c2492f",
        "high_pass": "#e0a030",
        "full": "#333333",
        "none": "#999999",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), dpi=160, sharey=True)
    for panel, correction in zip(axes, ("fail", "reflect")):
        for policy in POLICIES:
            series = [row for row in rows if row["policy"] == policy and row["correction"] == correction]
            if policy in ("none", "full"):
                value = series[-1]["mass_per_group"]
                panel.axhline(value, color=colours[policy], lw=1.0, ls=":", label=policy)
                continue
            series.sort(key=lambda row: row["budget_fraction"])
            panel.plot(
                [100 * row["label_fraction"] for row in series],
                [row["mass_per_group"] for row in series],
                marker="o", ms=3.5, lw=1.6, color=colours[policy], label=policy,
            )
        panel.axhline(0.0, color="#000000", lw=0.6)
        panel.set_xlabel("strong labels spent (% of a full audit)")
        panel.set_title("correction: %s" % correction, fontsize=10)
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("advantage left on hacks, per group")
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Where a limited strong verifier is worth spending", fontsize=11)
    figure.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    figure.savefig(path)
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
