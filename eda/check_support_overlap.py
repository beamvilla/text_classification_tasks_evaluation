"""Check that a support set does not copy or closely paraphrase the test set.

Usage (from the project root):
    uv run eda/check_support_overlap.py data/support_set_intent.csv data/review_shopping_intent.csv
    uv run eda/check_support_overlap.py data/support_set_sentiment.csv data/review_shopping_sentiment.csv --test-delimiter "\\t"

For every support sentence it reports the closest test sentence by two scores (1.0 = identical):
  - jaccard: overlap of character 3-grams (catches shared wording)
  - seq:     difflib SequenceMatcher ratio (catches near-identical order and wording)
Rows scoring above --max-seq or --max-jaccard are listed, and the exit code is 1 if there are any.
Both scores are lexical: they do not catch a paraphrase that uses different words.
"""
import argparse
import csv
import difflib
import sys


def read_column(path: str, column: str, delimiter: str) -> list[str]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.reader(f, delimiter=delimiter) if any(c.strip() for c in r)]
    header = rows[0]
    if len(header) == 1 and delimiter in header[0]:  # header written as one quoted cell
        header = header[0].split(delimiter)
    idx = header.index(column)
    return [r[idx].strip() for r in rows[1:] if len(r) > idx]


def grams(text: str, n: int = 3) -> set[str]:
    text = text.replace(" ", "")
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("support")
    ap.add_argument("test")
    ap.add_argument("--column", default="review")
    ap.add_argument("--support-delimiter", default=",")
    ap.add_argument("--test-delimiter", default=",")
    ap.add_argument("--max-seq", type=float, default=0.65)
    ap.add_argument("--max-jaccard", type=float, default=0.40)
    ap.add_argument("--top", type=int, default=5, help="how many closest pairs to print")
    args = ap.parse_args()

    support = read_column(args.support, args.column, args.support_delimiter.encode().decode("unicode_escape"))
    test = read_column(args.test, args.column, args.test_delimiter.encode().decode("unicode_escape"))
    test_grams = [(t, grams(t)) for t in test]

    results = []
    for s in support:
        g = grams(s)
        jac, jac_t = max((len(g & h) / len(g | h), t) for t, h in test_grams)
        seq, seq_t = max((difflib.SequenceMatcher(None, s.replace(" ", ""), t.replace(" ", "")).ratio(), t) for t in test)
        results.append((s, jac, seq, seq_t))

    exact = [s for s in support if s in set(test)]
    dups = len(support) - len(set(support))
    print(f"support={len(support)} test={len(test)} exact-matches={len(exact)} duplicates-within-support={dups}")
    print(f"max jaccard={max(r[1] for r in results):.2f}  max seq={max(r[2] for r in results):.2f}")
    print(f"\nTop {args.top} closest pairs (by seq):")
    for s, jac, seq, t in sorted(results, key=lambda r: -r[2])[: args.top]:
        print(f"  seq={seq:.2f} jaccard={jac:.2f}\n    support: {s}\n    test:    {t}")

    flagged = [r for r in results if r[2] > args.max_seq or r[1] > args.max_jaccard]
    print(f"\nFlagged (seq>{args.max_seq} or jaccard>{args.max_jaccard}): {len(flagged)}")
    for s, jac, seq, t in flagged:
        print(f"  {s}\n    ~ {t}  (seq={seq:.2f}, jaccard={jac:.2f})")
    return 1 if exact or dups or flagged else 0


if __name__ == "__main__":
    sys.exit(main())
