"""Train the toy policy under each audit policy and correction, and see what survives.

The static analysis in `advantage.py` says how much pull towards hacking each option leaves. This
asks whether that is what actually happens when you train on it, and what it costs in the only
thing that matters afterwards: how often the policy solves the task honestly.

    python experiments/train.py
    python experiments/train.py --budget 0.1 --steps 600
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lb_audit import train  # noqa: E402

ARMS = [
    ("none", "fail", "no audits at all"),
    ("random", "fail", "random groups, caught hacks count as failures"),
    ("low_pass", "fail", "fewest passes first, caught hacks count as failures"),
    ("random", "reflect", "random groups, caught hacks punished"),
    ("low_pass", "reflect", "fewest passes first, caught hacks punished"),
    ("random", "unbiased", "random groups, unbiased correction"),
    ("low_pass", "unbiased", "fewest passes first, unbiased correction"),
    ("full", "reflect", "audit everything (no budget)"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--solve-rate", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--out", default="results/train.json")
    parser.add_argument("--figure", default="figures/train.png")
    args = parser.parse_args()

    print("budget %.0f%% of a full audit, %d seeds, solve rate %.2f\n"
          % (100 * args.budget, args.seeds, args.solve_rate))
    print("%10s %10s %14s %16s %12s" % (
        "policy", "correction", "final hack rate", "held-out accuracy", "labels"))

    rows = []
    curves = {}
    for policy, correction, description in ARMS:
        runs = [
            train(
                policy=policy,
                correction=correction,
                budget_fraction=args.budget,
                steps=args.steps,
                batch=args.batch,
                group_size=args.group_size,
                solve_rate=args.solve_rate,
                learning_rate=args.lr,
                seed=seed,
            )
            for seed in range(args.seeds)
        ]
        final = [run.final for run in runs]
        row = {
            "policy": policy,
            "correction": correction,
            "description": description,
            "hack_rate": float(np.mean([f["hack_probability"] for f in final])),
            "hack_rate_sd": float(np.std([f["hack_probability"] for f in final])),
            "accuracy": float(np.mean([f["held_out_accuracy"] for f in final])),
            "accuracy_sd": float(np.std([f["held_out_accuracy"] for f in final])),
            "labels": float(np.mean([f["labels_used"] for f in final])),
            "update_variance": float(np.mean([f["update_variance"] for f in final])),
            "caught_reward": float(final[0]["caught_reward"]),
        }
        rows.append(row)
        key = "%s/%s" % (policy, correction)
        curves[key] = {
            "step": runs[0].column("step").tolist(),
            "hack": np.mean([run.column("hack_probability") for run in runs], axis=0).tolist(),
            "accuracy": np.mean([run.column("held_out_accuracy") for run in runs], axis=0).tolist(),
        }
        print("%10s %10s %13.3f   %14.3f %12.0f" % (
            policy, correction, row["hack_rate"], row["accuracy"], row["labels"]), flush=True)

    best = max(rows, key=lambda row: row["accuracy"])
    print("\nbest held-out accuracy: %s + %s at %.3f (ceiling is %.3f)"
          % (best["policy"], best["correction"], best["accuracy"], args.solve_rate))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"args": vars(args), "rows": rows, "curves": curves}, indent=2) + "\n")
        print("wrote %s" % args.out)
    if args.figure:
        plot(curves, rows, args)


def plot(curves, rows, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [
        ("none/fail", "no audits", "#999999", "-"),
        ("random/fail", "random + fail", "#e0a030", "-"),
        ("random/reflect", "random + reflect", "#c2492f", "-"),
        ("low_pass/reflect", "low-pass + reflect", "#2f6fc2", "-"),
        ("low_pass/unbiased", "low-pass + unbiased", "#7a3fa0", "--"),
        ("full/reflect", "full audit", "#333333", ":"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), dpi=160)
    for key, label, colour, style in shown:
        if key not in curves:
            continue
        steps = curves[key]["step"]
        axes[0].plot(steps, curves[key]["hack"], color=colour, ls=style, lw=1.6, label=label)
        axes[1].plot(steps, curves[key]["accuracy"], color=colour, ls=style, lw=1.6)
    axes[0].set_ylabel("probability of hacking")
    axes[0].set_title("does it learn to hack?", fontsize=10)
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[1].axhline(args.solve_rate, color="#000000", lw=0.7, ls=":")
    axes[1].set_ylabel("held-out accuracy")
    axes[1].set_title("what is it worth afterwards?", fontsize=10)
    for panel in axes:
        panel.set_xlabel("step")
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(alpha=0.25, linewidth=0.5)
    figure.suptitle(
        "GRPO with a strong verifier rationed to %.0f%% of a full audit" % (100 * args.budget),
        fontsize=11,
    )
    figure.tight_layout()
    os.makedirs(os.path.dirname(args.figure) or ".", exist_ok=True)
    figure.savefig(args.figure)
    print("wrote %s" % args.figure)


if __name__ == "__main__":
    main()
