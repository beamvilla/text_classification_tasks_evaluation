"""Plot the evaluation results in evaluated_results/all/summary.csv for the README.

Usage (from the repository root):
    uv run --with matplotlib eda/plot_evaluation_results.py [evaluated_results/<run>]

With no argument it reads evaluated_results/all/ (every run gathered in one summary.csv);
pass another folder to plot that one instead.

Writes assets/evaluation_accuracy_vs_time.png and assets/evaluation_cost.png (same look as
plot_class_distribution.py) and prints
the markdown tables used in README.md (accuracy / time / tokens, and cost from config/pricing.yaml).
"""
import csv
import random
import sys
from pathlib import Path

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "evaluated_results"
DEFAULT_RESULTS_DIR = RESULTS_ROOT / "all"  # every run gathered in one summary.csv
ASSETS = ROOT / "assets"
PRICING_PATH = ROOT / "config" / "pricing.yaml"
BASELINE = ("typesafe", "jev-latest")  # the run every other cost is compared with

# Playful red palette on a warm cream background, as in plot_class_distribution.py
BG, INK = "#FFF3E4", "#3A1F1D"
RED, YELLOW = "#E63946", "#FFC43D"
PROVIDER_COLORS = {"openai": YELLOW, "typesafe": RED}
PROVIDER_NAMES = {"openai": "OpenAI", "typesafe": "typesafe"}
# Not compared here: a few_shot run's speed depends on the machine its embedding server runs on
# (and it has no token counts or prices), so it cannot be ranked against the hosted APIs.
EXCLUDED_PROVIDERS = {"few_shot"}
USECASE_TITLES = {"intent": "Intent", "sentiment": "Sentiment"}
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


# ---------------------------------------------------------------- data
def load_rows(results_dir: Path) -> list[dict]:
    with open(results_dir / "summary.csv", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["provider"] not in EXCLUDED_PROVIDERS]
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
# Labels are placed automatically, so new runs need no manual layout. Each label has a set of candidate offsets
# (in points) around its dot; the layout with the fewest collisions wins (labels on labels or dots, outside the
# panel, leader lines through other dots or labels), preferring labels close to their dot. Labels further away
# than the first ring get a thin leader line back to their dot.
MARKER_SIZE = 110
LABEL_FONTSIZE = 8
NEAR_OFFSETS = [(0, 8), (0, -8), (8, 0), (-8, 0), (6, 6), (-6, 6), (6, -6), (-6, -6)]
FAR_OFFSETS = [(dx, s * d) for d in (26, 44, 62, 80, 98, 116) for s in (1, -1) for dx in (0, -30, 30)]
HEADER_Y = 0.765  # labels may rise above a panel up to here (figure fraction), just under the panel title


def _crosses(segment, box: Bbox) -> bool:
    (x0, y0), (x1, y1) = segment
    return any(box.contains(x0 + (x1 - x0) * t / 24, y0 + (y1 - y0) * t / 24) for t in range(25))


def place_labels(ax, points: list[tuple[float, float, str]]) -> None:
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    panel = ax.get_window_extent(renderer)
    top = fig.transFigure.transform((0, HEADER_Y))[1]
    pad = 2  # px of air around every label
    radius = MARKER_SIZE ** 0.5 / 2 / 72 * fig.dpi
    centres = ax.transData.transform([(x, y) for x, y, _ in points])
    dots = [Bbox.from_extents(px - radius, py - radius, px + radius, py + radius) for px, py in centres]
    style = dict(textcoords="offset points", fontsize=LABEL_FONTSIZE, fontweight="bold", linespacing=1.25,
                 annotation_clip=False, zorder=4)

    def label(i, dx, dy, leader=False):
        x, y, text = points[i]
        ha = "center" if dx == 0 else ("left" if dx > 0 else "right")
        va = "center" if dy == 0 else ("bottom" if dy > 0 else "top")
        extra = dict(arrowprops=dict(arrowstyle="-", color=INK, lw=1, alpha=0.6, shrinkA=2,
                                     shrinkB=radius * 72 / fig.dpi + 1)) if leader else {}
        return ax.annotate(text, (x, y), xytext=(dx, dy), ha=ha, va=va, **style, **extra)

    # every candidate of every label: (offset, box, leader segment or None, cost on its own)
    options = []
    for i, (px, py) in enumerate(centres):
        mine = []
        for rank, (dx, dy) in enumerate(NEAR_OFFSETS + FAR_OFFSETS):
            ann = label(i, dx, dy)
            b = ann.get_window_extent(renderer)
            ann.remove()
            box = Bbox.from_extents(b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad)
            far = (dx, dy) in FAR_OFFSETS
            segment = ((min(max(px, box.x0), box.x1), min(max(py, box.y0), box.y1)), (px, py)) if far else None
            cost = rank * 0.1 + (abs(dx) + abs(dy)) * (1.0 if far else 0.2)
            if not (panel.x0 <= box.x0 and box.x1 <= panel.x1 and panel.y0 <= box.y0 and box.y1 <= top):
                cost += 1e6
            cost += 1e4 * sum(box.overlaps(d) for d in dots)
            # a label without a leader line must have its own dot closer to its centre than any other dot,
            # or it reads as belonging to that other dot
            if not far:
                cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
                own = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                cost += 1e3 * sum(((cx - qx) ** 2 + (cy - qy) ** 2) ** 0.5 < own
                                  for j, (qx, qy) in enumerate(centres) if j != i)
            if far:
                cost += 1e3 * sum(_crosses(segment, d) for j, d in enumerate(dots) if j != i)
            mine.append(((dx, dy), box, segment, cost))
        options.append(mine)

    def pair_cost(a, b) -> float:
        cost = 1e4 if a[1].overlaps(b[1]) else 0.0
        for one, other in ((a, b), (b, a)):
            if one[2] is not None and _crosses(one[2], other[1]):
                cost += 1e3
        return cost

    def improve(choice: list[int]) -> list[int]:
        for _ in range(30):  # one label at a time takes its best candidate, until nothing changes
            changed = False
            for i in range(len(points)):
                def total(k):
                    return options[i][k][3] + sum(pair_cost(options[i][k], options[j][choice[j]])
                                                  for j in range(len(points)) if j != i)
                best = min(range(len(options[i])), key=total)
                if total(best) < total(choice[i]) - 1e-9:
                    choice[i], changed = best, True
            if not changed:
                break
        return choice

    def layout_cost(choice: list[int]) -> float:
        n = len(points)
        return (sum(options[i][choice[i]][3] for i in range(n))
                + sum(pair_cost(options[i][choice[i]], options[j][choice[j]]) for i in range(n) for j in range(i + 1, n)))

    rng = random.Random(0)  # fixed seed: the same results always give the same chart
    starts = [[min(range(len(o)), key=lambda k: o[k][3]) for o in options]]
    starts += [[rng.randrange(len(o)) for o in options] for _ in range(40)]
    best = min((improve(c) for c in starts), key=layout_cost)
    for i, k in enumerate(best):
        (dx, dy), _, segment, _ = options[i][k]
        label(i, dx, dy, leader=segment is not None)


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
        seconds = range(int(xmax) + 1)  # one tick per whole second that fits, however slow the slowest run
        ax.set_xticks(list(seconds))
        ax.set_xticklabels([f"{t} s" for t in seconds], fontsize=12, fontweight="bold")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=INK, alpha=0.12, linewidth=1)
        ax.axhline(0.83, color=INK, linewidth=2, zorder=2)
        for r in data:
            ax.scatter([r["mean_inference_time_s"]], [r["accuracy"]], s=MARKER_SIZE,
                       color=PROVIDER_COLORS[r["provider"]], edgecolors=INK, linewidths=1.3, zorder=3)
        place_labels(ax, [(r["mean_inference_time_s"], r["accuracy"], f"{r['model']}\n{r['setting']}")
                          for r in data])

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


# ---------------------------------------------------------------- chart: cost
def plot_cost(rows: list[dict], path: Path) -> None:
    """Total cost of one evaluation run per model, one panel per usecase, same x scale in both."""
    n = len(by_usecase(rows, "intent"))
    # The figure grows with the number of bars so every two-line model label keeps 0.5 in of room. The header and
    # footer keep their size in inches; top() and bottom() turn a position on the 6.8 in tall design into a
    # figure fraction measured from the top or the bottom edge.
    W, BASE_H = 13, 6.8
    panel_h = max(0.60 * BASE_H, 0.5 * (n + 0.2))
    H = BASE_H - 0.60 * BASE_H + panel_h

    def top(y: float) -> float:
        return 1 - (1 - y) * BASE_H / H

    def bottom(y: float) -> float:
        return y * BASE_H / H

    fig = plt.figure(figsize=(W, H), facecolor=BG)
    width = 0.34
    layout = {"intent": [0.16, bottom(0.12), width, panel_h / H],
              "sentiment": [0.58, bottom(0.12), width, panel_h / H]}
    xmax = max(r["total_cost"] for r in rows) * 1.7

    for usecase, box in layout.items():
        ax = fig.add_axes(box)
        data = by_usecase(rows, usecase)
        base = next(r["total_cost"] for r in data if (r["provider"], r["model"]) == BASELINE)
        ax.set_facecolor(BG)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xticks([])
        ax.axvline(0, color=INK, linewidth=2)
        ax.tick_params(length=0)
        px_x, px_y = W * box[2] / xmax, H * box[3] / (n + 0.2)  # inches per data unit
        radius = 0.09 / px_x  # corner radius of 0.09 in, in x data units
        for i, r in enumerate(data):
            y = n - 1 - i
            w = r["total_cost"]
            color = PROVIDER_COLORS[r["provider"]]
            if w > 2 * radius:
                ax.add_patch(FancyBboxPatch((0, y - 0.3), w, 0.6, boxstyle=f"round,pad=0,rounding_size={radius}",
                                            mutation_aspect=px_x / px_y, fc=color, ec=INK, lw=2))
            else:  # too thin to round; keep the outline so it stays visible
                ax.add_patch(Rectangle((0, y - 0.3), w, 0.6, fc=color, ec=INK, lw=2))
            label = f"${w:.4f}" if (r["provider"], r["model"]) == BASELINE else f"${w:.4f}  ({w / base:.1f}x)"
            ax.text(w + xmax * 0.025, y, label, va="center", fontsize=13, fontweight="bold")
        if usecase == "intent":
            ax.set_yticks(range(n))
            ax.set_yticklabels([f"{r['model']}\n{r['setting']}" for r in data][::-1], fontsize=12, fontweight="bold",
                               linespacing=1.25)
        else:
            ax.set_yticks([])
        fig.text(box[0], top(0.78), f"{usecase.upper()}  n={data[0]['samples']}", fontsize=13, fontweight="bold")

    fig.text(0.54, bottom(0.04), "Total cost of one evaluation run (USD); (Nx) = times the jev-latest cost",
             fontsize=13, fontweight="bold", ha="center")
    for j, provider in enumerate(dict.fromkeys(r["provider"] for r in rows)):
        x = 0.68 + j * 0.13
        fig.patches.append(FancyBboxPatch((x, top(0.895)), 0.022, 0.035 * BASE_H / H, transform=fig.transFigure,
                           boxstyle="round,pad=0,rounding_size=0.008", mutation_aspect=H / W,
                           fc=PROVIDER_COLORS[provider], ec=INK, lw=2))
        fig.text(x + 0.03, top(0.912), PROVIDER_NAMES[provider], va="center", fontsize=13, fontweight="bold")
    fig.text(0.06, top(0.92), "Cost Benchmark", fontsize=28, fontweight="bold", color=RED)
    fig.text(0.06, top(0.86), "Same samples and prompts for every model: lower is cheaper", fontsize=13)
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
            r["input_cost"] = r["output_cost"] = r["total_cost"] = None
            continue
        r["input_cost"] = r["input_tokens"] / 1_000_000 * price[0]
        r["output_cost"] = r["output_tokens"] / 1_000_000 * price[1]
        r["total_cost"] = r["input_cost"] + r["output_cost"]


def usd(value, digits=6) -> str:
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
        base = next((r["total_cost"] for r in data if (r["provider"], r["model"]) == BASELINE), None)
        print(f"\n**{USECASE_TITLES[usecase]}** ({data[0]['samples']} samples)\n")
        print("| Model | Setting | Accuracy | Input tokens | Output tokens | Reasoning tokens | Input cost | Output cost | Total cost | jev-latest is cheaper by |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in data:
            reasoning = "n/a" if r["reasoning_tokens"] is None else f"{r['reasoning_tokens']:,}"
            if (r["provider"], r["model"]) == BASELINE:
                ratio = "-"
            elif r["total_cost"] is None or not base:
                ratio = "n/a"
            else:
                times = r["total_cost"] / base
                ratio = f"{times:.1f}x"
            print(f"| `{r['model']}` | {r['setting']} | {r['accuracy']:.1%} | {r['input_tokens']:,} | {r['output_tokens']:,} | "
                  f"{reasoning} | {usd(r['input_cost'])} | {usd(r['output_cost'])} | {usd(r['total_cost'])} | {ratio} |")


def main() -> None:
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1]).resolve()
    else:
        results_dir = DEFAULT_RESULTS_DIR
    rows = load_rows(results_dir)
    pricing = load_pricing()
    add_costs(rows, pricing)
    plot_accuracy_vs_time(rows, ASSETS / "evaluation_accuracy_vs_time.png")
    plot_cost(rows, ASSETS / "evaluation_cost.png")
    print(f"Read {results_dir.relative_to(ROOT)}; wrote assets/evaluation_accuracy_vs_time.png and assets/evaluation_cost.png")
    print_tables(rows)
    print_cost_tables(rows, pricing)


if __name__ == "__main__":
    main()
