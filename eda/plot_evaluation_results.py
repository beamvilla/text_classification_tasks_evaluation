"""Plot the evaluation results in evaluated_results/<run>/summary.csv for the README.

Usage (from the repository root):
    uv run --with matplotlib eda/plot_evaluation_results.py [evaluated_results/<run>]

Writes assets/evaluation_accuracy_vs_time.png (same look as plot_class_distribution.py) and prints
the markdown tables used in README.md (accuracy / time / tokens, and cost from config/pricing.yaml).
"""
import csv
import sys
from pathlib import Path

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "evaluated_results"
ASSETS = ROOT / "assets"
PRICING_PATH = ROOT / "config" / "pricing.yaml"
BASELINE = ("typesafe", "jev-latest")  # the run every other cost is compared with

# Playful red palette on a warm cream background, as in plot_class_distribution.py
BG, INK = "#FFF3E4", "#3A1F1D"
RED, YELLOW = "#E63946", "#FFC43D"
PROVIDER_COLORS = {"openai": YELLOW, "typesafe": RED}
PROVIDER_NAMES = {"openai": "OpenAI", "typesafe": "typesafe"}
USECASE_TITLES = {"intent": "Intent", "sentiment": "Sentiment"}
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


# ---------------------------------------------------------------- data
def load_rows(results_dir: Path) -> list[dict]:
    with open(results_dir / "summary.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in ("accuracy", "macro_precision", "macro_recall", "mean_inference_time_s"):
            r[key] = float(r[key])
        for key in ("samples", "input_tokens", "output_tokens"):
            r[key] = int(r[key])
        r["reasoning_tokens"] = int(r["reasoning_tokens"]) if r["reasoning_tokens"] != "" else None
        r["setting"] = setting(r)
        r["label"] = f"{r['model']} · {r['setting']}"
    return rows


def setting(r: dict) -> str:
    if r["reasoning_effort"]:
        return r["reasoning_effort"]
    if r["temperature"]:
        return f"temperature {float(r['temperature']):g}"
    return "typesafe" if r["provider"] == "typesafe" else "default"


def by_usecase(rows: list[dict], usecase: str) -> list[dict]:
    return [r for r in rows if r["usecase"] == usecase]


# ---------------------------------------------------------------- chart: accuracy vs time
# Which side of its point each label goes, so labels of close points never collide.
LABEL_SIDE = {
    "intent": {"gpt-4.1-mini · temperature 0": -1, "gpt-5-mini · minimal": -1, "gpt-5-mini · medium": 1,
               "gpt-5 · medium": 1, "jev-latest · typesafe": 1},
    "sentiment": {"gpt-4.1-mini · temperature 0": -1, "gpt-5-mini · minimal": 1, "gpt-5-mini · medium": -1,
                  "gpt-5 · medium": 1, "jev-latest · typesafe": 1},
}


def plot_accuracy_vs_time(rows: list[dict], path: Path) -> None:
    W, H = 13, 6.8
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    panels = {"intent": fig.add_axes([0.075, 0.15, 0.40, 0.58]),
              "sentiment": fig.add_axes([0.555, 0.15, 0.40, 0.58])}
    xmax = max(r["mean_inference_time_s"] for r in rows) * 1.18

    for usecase, ax in panels.items():
        data = by_usecase(rows, usecase)
        ax.set_facecolor(BG)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=0, labelsize=12)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0.83, 1.05)
        ax.set_yticks([0.85, 0.90, 0.95, 1.00])
        ax.set_yticklabels(["85%", "90%", "95%", "100%"], fontsize=12, fontweight="bold")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["0 s", "1 s", "2 s", "3 s"], fontsize=12, fontweight="bold")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=INK, alpha=0.12, linewidth=1)
        ax.axhline(0.83, color=INK, linewidth=2, zorder=2)
        for r in data:
            x, y = r["mean_inference_time_s"], r["accuracy"]
            ax.scatter([x], [y], s=330, color=PROVIDER_COLORS[r["provider"]], edgecolors=INK,
                       linewidths=2, zorder=3)
            side = LABEL_SIDE[usecase][r["label"]]
            ax.annotate(f"{r['model']}\n{r['setting']}", (x, y), xytext=(0, 17 * side),
                        textcoords="offset points", ha="center", va="bottom" if side > 0 else "top",
                        fontsize=10.5, fontweight="bold", linespacing=1.25)

    n = {u: by_usecase(rows, u)[0]["samples"] for u in panels}
    perfect = all(r["accuracy"] == 1 for r in by_usecase(rows, "sentiment"))
    fig.text(0.075, 0.78, f"INTENT  n={n['intent']}", fontsize=13, fontweight="bold")
    fig.text(0.555, 0.78, f"SENTIMENT  n={n['sentiment']}" + ("  ·  every run scores 100%" if perfect else ""),
             fontsize=13, fontweight="bold")
    panels["intent"].set_ylabel("Accuracy", fontsize=13, fontweight="bold", labelpad=10)
    fig.text(0.515, 0.04, "Mean inference time per sample", fontsize=13, fontweight="bold", ha="center")

    # legend chips, as in plot_class_distribution.py
    for j, provider in enumerate(dict.fromkeys(r["provider"] for r in rows)):
        x = 0.68 + j * 0.13
        fig.patches.append(FancyBboxPatch((x, 0.895), 0.022, 0.035, transform=fig.transFigure,
                           boxstyle="round,pad=0,rounding_size=0.008", fc=PROVIDER_COLORS[provider], ec=INK, lw=2))
        fig.text(x + 0.03, 0.912, PROVIDER_NAMES[provider], va="center", fontsize=13, fontweight="bold")

    fig.text(0.06, 0.92, "Accuracy vs Inference Time", fontsize=28, fontweight="bold", color=RED)
    fig.text(0.06, 0.86, "One point per run: higher and further left is better", fontsize=13)
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------- markdown tables for the README
def print_tables(rows: list[dict]) -> None:
    for usecase in ("intent", "sentiment"):
        data = by_usecase(rows, usecase)
        print(f"\n**{USECASE_TITLES[usecase]}** ({data[0]['samples']} samples)\n")
        print("| Provider | Model | Setting | Accuracy | Macro precision | Macro recall | Mean time (s) | Output tokens / sample |")
        print("|---|---|---|---|---|---|---|---|")
        for r in data:
            out = r["output_tokens"] / r["samples"]
            print(f"| {PROVIDER_NAMES[r['provider']]} | `{r['model']}` | {r['setting']} | {r['accuracy']:.1%} | "
                  f"{r['macro_precision']:.1%} | {r['macro_recall']:.1%} | {r['mean_inference_time_s']:.2f} | {out:.0f} |")


# ---------------------------------------------------------------- cost tables
def load_pricing() -> dict:
    """{(provider, model): (input_usd_per_1m, output_usd_per_1m)} from config/pricing.yaml."""
    with open(PRICING_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {(provider, model): (float(p["input"]), float(p["output"]))
            for provider, models in raw.items() for model, p in models.items()}


def add_costs(rows: list[dict], pricing: dict) -> None:
    """cost = tokens / 1,000,000 * price per 1M tokens. OpenAI bills reasoning tokens as output tokens,
    and they are already part of `output_tokens`. Rows without a price get None."""
    for r in rows:
        price = pricing.get((r["provider"], r["model"]))
        if price is None:
            r["input_cost"] = r["output_cost"] = r["total_cost"] = r["cost_per_1000"] = None
            continue
        r["input_cost"] = r["input_tokens"] / 1_000_000 * price[0]
        r["output_cost"] = r["output_tokens"] / 1_000_000 * price[1]
        r["total_cost"] = r["input_cost"] + r["output_cost"]
        r["cost_per_1000"] = r["total_cost"] / r["samples"] * 1000


def usd(value, digits=4) -> str:
    if value is None:
        return "n/a"
    return "$0" if value == 0 else f"${value:.{digits}f}"


def price(value: float) -> str:
    """$0.25, $2.00, $0.042, $0 - at least two decimals, at most three."""
    if value == 0:
        return "$0"
    text = f"{value:.3f}"
    return "$" + (text[:-1] if text.endswith("0") else text)


def print_cost_tables(rows: list[dict], pricing: dict) -> None:
    print("\n**Prices** (USD per 1M tokens)\n")
    print("| Provider | Model | Input | Output |")
    print("|---|---|---|---|")
    seen = dict.fromkeys((r["provider"], r["model"]) for r in rows)
    for provider, model in seen:
        p = pricing.get((provider, model))
        cells = ("n/a", "n/a") if p is None else (price(p[0]), price(p[1]))
        print(f"| {PROVIDER_NAMES[provider]} | `{model}` | {cells[0]} | {cells[1]} |")
    for usecase in ("intent", "sentiment"):
        data = by_usecase(rows, usecase)
        base = next((r["cost_per_1000"] for r in data if (r["provider"], r["model"]) == BASELINE), None)
        print(f"\n**{USECASE_TITLES[usecase]}** ({data[0]['samples']} samples)\n")
        print("| Model | Setting | Accuracy | Input tokens | Output tokens | Input cost | Output cost | Total cost | Per 1,000 samples | jev-latest is cheaper by |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in data:
            if (r["provider"], r["model"]) == BASELINE:
                ratio = "-"
            elif r["cost_per_1000"] is None or not base:
                ratio = "n/a"
            else:
                times = r["cost_per_1000"] / base
                ratio = f"{times:.1f}x ({1 - 1 / times:.1%} less)"
            print(f"| `{r['model']}` | {r['setting']} | {r['accuracy']:.1%} | {r['input_tokens']:,} | {r['output_tokens']:,} | "
                  f"{usd(r['input_cost'])} | {usd(r['output_cost'])} | {usd(r['total_cost'])} | {usd(r['cost_per_1000'])} | {ratio} |")


def main() -> None:
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1]).resolve()
    else:
        results_dir = sorted(p for p in RESULTS_ROOT.iterdir() if (p / "summary.csv").exists())[-1]
    rows = load_rows(results_dir)
    plot_accuracy_vs_time(rows, ASSETS / "evaluation_accuracy_vs_time.png")
    print(f"Read {results_dir.relative_to(ROOT)}; wrote assets/evaluation_accuracy_vs_time.png")
    print_tables(rows)
    pricing = load_pricing()
    add_costs(rows, pricing)
    print_cost_tables(rows, pricing)


if __name__ == "__main__":
    main()
