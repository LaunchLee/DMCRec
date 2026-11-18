"""Entry point for the revised DMCRec study; default action only lists groups.

Example: python main.py --group Baby/baby_primary/DMCRec/full
Training requires an explicit --execute flag. Use --data-dir datasets when
using the existing repository-root dataset directory.
"""
import sys
from experiments.dmcrec.tools.reproduce_paper import main

if __name__ == "__main__":
    main(sys.argv[1:] or ["--list"])
