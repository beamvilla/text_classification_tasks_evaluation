"""Generate a playful red-themed class-distribution infographic for data/."""
import csv
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Playful red palette on a warm cream background
BG, INK = "#FFF3E4", "#3A1F1D"
RED, CORAL, YELLOW, PINK = "#E63946", "#FF7A59", "#FFC43D", "#FF9FB2"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

INTENTS = ["order_status", "return_refund", "product_question",
           "payment_issue", "account_issue", "complaint", "other"]

# ---- load data ----
sent = Counter()
with open("data/review_shopping_sentiment.csv", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('"review'):
            sent[line.rsplit("\t", 1)[1]] += 1
real, syn = Counter(), Counter()
with open("data/review_shopping_intent.csv", encoding="utf-8", errors="replace") as f:
    for r in csv.DictReader(f):
        (real if r["source"] == "real" else syn)[r["intent"]] += 1

# ---- layout ----
W, H = 13, 6.8
fig = plt.figure(figsize=(W, H), facecolor=BG)
L1 = [0.06, 0.14, 0.25, 0.62]   # vertical chart (sentiment)
L2 = [0.46, 0.14, 0.46, 0.62]   # horizontal chart (intent)
a1, a2 = fig.add_axes(L1), fig.add_axes(L2)

def rbar(ax, x, y, w, h, color, r, aspect):
    """Rounded bar in data coordinates; `aspect` keeps the corners circular."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=aspect, fc=color, ec=INK, lw=2))

def style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

# ---- vertical bars: sentiment ----
style(a1)
ymax = 80
a1.set_xlim(-0.6, 1.6); a1.set_ylim(0, ymax)
px_x = (W * L1[2]) / 2.2; px_y = (H * L1[3]) / ymax
for i, (lab, col) in enumerate([("pos", PINK), ("neg", RED)]):
    v = sent[lab]
    rbar(a1, i - 0.3, 0, 0.6, v, col, 0.25, px_x / px_y)
    a1.text(i, v + 1.5, str(v), ha="center", fontsize=20, fontweight="bold")
a1.set_xticks([0, 1]); a1.set_xticklabels(["pos", "neg"], fontsize=14, fontweight="bold")
a1.set_yticks([]); a1.axhline(0, color=INK, lw=2)

# ---- horizontal bars: intent ----
style(a2)
xmax, n = 24, len(INTENTS)
a2.set_xlim(0, xmax); a2.set_ylim(-0.6, n - 0.4)
px_x = (W * L2[2]) / xmax; px_y = (H * L2[3]) / (n + 0.2)
for i, k in enumerate(INTENTS):
    y = n - 1 - i
    r, s = real[k], syn[k]
    if s:   # synthetic part drawn full length, real part layered on top
        rbar(a2, 0, y - 0.3, r + s, 0.6, RED, 0.3, px_x / px_y)
    if r:
        rbar(a2, 0, y - 0.3, r if s else r, 0.6, YELLOW, 0.3, px_x / px_y)
    a2.text(r + s + 0.5, y, str(r + s), va="center", fontsize=15, fontweight="bold")
a2.set_yticks(range(n)); a2.set_yticklabels(INTENTS[::-1], fontsize=13, fontweight="bold")
a2.set_xticks([]); a2.axvline(0, color=INK, lw=2)

# legend chips
for j, (lab, col) in enumerate([("real", YELLOW), ("synthetic", RED)]):
    fig.patches.append(FancyBboxPatch((0.66 + j * 0.13, 0.80), 0.022, 0.035, transform=fig.transFigure,
                       boxstyle="round,pad=0,rounding_size=0.008", fc=col, ec=INK, lw=2))
    fig.text(0.69 + j * 0.13, 0.817, lab, va="center", fontsize=13, fontweight="bold")

# titles
fig.text(0.06, 0.92, "Class Distribution", fontsize=28, fontweight="bold", color=RED)
fig.text(0.06, 0.86, "Thai shopping reviews: sentiment and customer intent", fontsize=13)
fig.text(0.06, 0.78, f"SENTIMENT  n={sum(sent.values())}", fontsize=13, fontweight="bold")
fig.text(0.46, 0.78, f"INTENT  n={sum(real.values()) + sum(syn.values())}", fontsize=13, fontweight="bold")

fig.savefig("assets/class_distribution.png", dpi=150, facecolor=BG)
print(dict(sent), dict(real), dict(syn))
