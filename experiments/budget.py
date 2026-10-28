"""How big does the audit budget have to be before hacking stops?

`advantage.py` predicts a threshold: the budget at which the advantage left on hacks reaches zero.
Below it the policy gradient still points towards hacking and training walks there; above it the
pull reverses. This runs the training loop across budgets and checks whether the threshold the
static analysis predicted is where behaviour actually turns over.

    python experiments/budget.py
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lb_audit import train  # noqa: E402

ARMS = [
    ("random", "fail"),
    ("low_pass", "fail"),
    ("random", "reflect"),
    ("low_pass", "reflect"),
    ("random", "unbiased"),
    ("low_pass", "unbiased"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.325, 0.35, 0.375, 0.4, 0.5, 0.7])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--solve-rate", type=float, default=0.35)
    parser.add_argument("--out", default="results/budget.json")
    parser.add_argument("--figure", default="figures/budget.png")
    args = parser.parse_args()

    labels = ["%s/%s" % arm for arm in ARMS]
    print("final hack rate after %d steps, %d seeds\n" % (args.steps, args.seeds))
    print("%8s  %s" % ("budget", "  ".join("%18s" % name for name in labels)))

    rows = []
    for budget in args.budgets:
        line = []
        for policy, correction in ARMS:
            finals = []
            for seed in range(args.seeds):
                run = train(
                    policy=policy,
                    correction=correction,
                    budget_fraction=budget,
                    steps=args.steps,
                    solve_rate=args.solve_rate,
                    seed=seed,
                )
                finals.append(run.final)
            row = {
                "budget": budget,
                "policy": policy,
                "correction": correction,
                "hack_rate": float(np.mean([f["hack_probability"] for f in finals])),
                "accuracy": float(np.mean([f["held_out_accuracy"] for f in finals])),
                "update_variance": float(np.mean([f["update_variance"] for f in finals])),
                "labels": float(np.mean([f["labels_used"] for f in finals])),
            }
            rows.append(row)
            line.append(row["hack_rate"])
        print("%7.0f%%  %s" % (100 * budget, "  ".join("%18.3f" % value for value in line)),
              flush=True)

    print("\nsmallest budget that leaves the final hack rate under 10%:")
    for policy, correction in ARMS:
        series = [r for r in rows if r["policy"] == policy and r["correction"] == correction]
        series.sort(key=lambda r: r["budget"])
        hit = next((r for r in series if r["hack_rate"] < 0.1), None)
        print("  %-18s %s" % (
            "%s/%s" % (policy, correction),
            "%.0f%%" % (100 * hit["budget"]) if hit else "never, in this range"))

    print("")
    print("(for how noisy each individual step is, see experiments/variance.py,")
    print(" which measures it at a policy that is not moving.)")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"args": vars(args), "rows": rows}, indent=2) + "\n")
        print("wrote %s" % args.out)
    if args.figure:
        plot(rows, args)


def plot(rows, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        ("random", "fail"): ("#e0a030", "--", "random + fail"),
        ("low_pass", "fail"): ("#e0a030", "-", "low-pass + fail"),
        ("random", "reflect"): ("#c2492f", "--", "random + reflect"),
        ("low_pass", "reflect"): ("#2f6fc2", "-", "low-pass + reflect"),
        ("random", "unbiased"): ("#7a3fa0", "--", "random + unbiased"),
        ("low_pass", "unbiased"): ("#7a3fa0", "-", "low-pass + unbiased"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), dpi=160)
    for arm, (colour, style, label) in styles.items():
        series = [r for r in rows if (r["policy"], r["correction"]) == arm]
        series.sort(key=lambda r: r["budget"])
        xs = [100 * r["budget"] for r in series]
        axes[0].plot(xs, [r["hack_rate"] for r in series], color=colour, ls=style,
                     lw=1.6, marker="o", ms=3, label=label)
        axes[1].plot(xs, [r["accuracy"] for r in series], color=colour, ls=style,
                     lw=1.6, marker="o", ms=3)
    axes[0].set_ylabel("final probability of hacking")
    axes[0].set_title("does it stop hacking?", fontsize=10)
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[1].axhline(args.solve_rate, color="#000000", lw=0.7, ls=":")
    axes[1].set_ylabel("held-out accuracy")
    axes[1].set_title("what is it worth afterwards?", fontsize=10)
    for panel in axes:
        panel.set_xlabel("strong-label budget (% of a full audit)")
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(alpha=0.25, linewidth=0.5)
    figure.suptitle("How much strong verification is enough", fontsize=11)
    figure.tight_layout()
    os.makedirs(os.path.dirname(args.figure) or ".", exist_ok=True)
    figure.savefig(args.figure)
    print("wrote %s" % args.figure)


if __name__ == "__main__":
    main()
