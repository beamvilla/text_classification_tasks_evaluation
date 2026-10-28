from dotenv import load_dotenv
load_dotenv()

import argparse
import logging
import sys
sys.path.append("./")

from src.config.config import DEFAULT_CONFIG_PATH, load_eval_config
from src.evaluation.runner import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate text classification models (see config/config.yaml).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to the YAML config")
    parser.add_argument("--dry-run", action="store_true", help="validate the config and show the plan without calling any API")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_eval_config(args.config)
    output_dir, rows = run_evaluation(config, dry_run=args.dry_run)
    if args.dry_run:
        return

    print(f"\nResults written to {output_dir}\n")
    header = f"{'usecase':<10} {'provider:model':<28} {'acc':>6} {'prec':>6} {'recall':>6} {'errors':>6} {'in_tok':>8} {'out_tok':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['usecase']:<10} {r['provider'] + ':' + r['model']:<28} {r['accuracy']!s:>6} {r['macro_precision']!s:>6} {r['macro_recall']!s:>6} "
              f"{r['errors']:>6} {r['input_tokens']!s:>8} {r['output_tokens']!s:>8}")


if __name__ == "__main__":
    main()
