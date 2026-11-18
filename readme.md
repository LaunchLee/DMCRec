# DMCRec: Dual-Stream Multimodal Graph Network with Counterfactual-Inspired Scoring for Recommendation

This release supports the revised manuscript on multimodal e-commerce product recommendation. The model combines dual-stream graph propagation, cross-view alignment, loss-plateau-gated topology refresh and matching/item-modulation training. Its scoring intervention is defined on a fitted model. The results characterize overall-accuracy versus tail-item/coverage trade-offs on Amazon Baby and Sports.

## What to run

The corrected implementation is under `experiments/dmcrec/`. The original root `main.py` now calls the revised reproduction runner and lists available groups when run without arguments. Root `data.py`, `misc.py` and `my_models/dmcrec.py` are compatibility imports pointing to the shared corrected modules, rather than independent historical implementations. Previous source remains in Git history.

- `experiments/dmcrec/tools/reproduce_paper.py`: preview or replay any of the 23 frozen selected experimental groups.
- `experiments/dmcrec/configs/paper_configurations.json`: exact recorded constructor settings and scoring modes.
- `research_results/`: derived seed-level metrics, data protocol/fingerprints, full-precision summaries and a CPU-only verification script.

```bash
python research_results/verify_results.py
python main.py  # List groups without loading data or training.
python main.py --group Baby/baby_primary/DMCRec/full --data-dir datasets
python experiments/dmcrec/tools/reproduce_paper.py --list
python experiments/dmcrec/tools/reproduce_paper.py --group Baby/baby_primary/DMCRec/full
```

The last command only previews configuration and command text. Add `--execute` to train. To check feasibility first, use `--smoke --seeds 42 --execute`; the two-epoch, 256-item output is separate from the manuscript results. Full DMCRec training uses 200 epochs and batch 8192, with five seeds by default. Choose `--device cuda` for GPU graph training, and pass `--data-dir` and `--output-dir` explicitly. The MF-BCE/MACR-MF extension uses the recorded single-thread CPU protocol and requires `--device cpu`.

```bash
python experiments/dmcrec/tools/reproduce_paper.py --group Baby/baby_primary/DMCRec/full --device cuda --data-dir /path/to/datasets --output-dir /path/to/new_runs --seeds 42 --smoke --execute
```

The Google Drive folder in `datasets/readme.md` shares datasets used in the earlier experiments. That dataset link is retained separately from the revised aggregate results. Reproduction uses the Python entry points above; a Colab notebook is not required.

## Environment and data

The recorded environment used Python 3.11, the dependencies in `experiments/dmcrec/requirements-remote.txt`, and separately installed GPU-compatible PyTorch. Inspect compatibility before installing in a different environment. The Python entry point has passed local CPU smoke runs. Record software versions, source commit and hardware for any replication.

Get Baby and Sports interaction/feature files from the dataset links maintained by [MMRec](https://github.com/enoche/MMRec). Place `MMRec-Baby/` and `MMRec-Sports/` under one common dataset directory. When using the existing root `datasets/` directory, pass `--data-dir datasets` explicitly (see `datasets/readme.md`). Required files are `baby.inter` or `sports.inter`, `image_feat.npy`, `text_feat.npy`, `i_id_mapping.csv`, and `u_id_mapping.csv`. Preserve supplied identity mappings and feature-row alignment. Third-party source data and large feature arrays are not redistributed in this release. Their source-provider terms apply.

The revised study treats all observed ratings as binary feedback and uses a random 70/10/20 split with split seed 42, rather than the source x_label assignments. Unknown held-out entities are excluded. The included data report records exact counts and input fingerprints. Do not combine these results with older filtered-data tables or the discarded batch-512 experiments.

## Scope and result provenance

The archive records 115 final model evaluations in 23 five-seed groups. DMCRec/WMF/MENTOR use two-candidate validation-only selection; component and refresh controls inherit the full model's selected configuration. DMCRec contrastive weights are 0.005 on Baby and 0.01 on Sports. The independent MF-BCE/MACR-MF adaptation uses its own recorded learning-rate/reference selection; it is an explicitly adapted 200-epoch comparison. Seed42 is used in selection and final summaries. Training and evaluation samplers remain model-specific.

The entry point replays the frozen selected configurations; it does not repeat hyperparameter selection. Newly produced metrics are separate replications and never overwrite `research_results`. CUDA sparse operations, libraries and hardware can affect numerical results. Five training seeds on one split describe conditional training variability. The reported scoring interventions characterize a learned prediction function.

## Attribution

The implementation uses [Cornac](https://github.com/PreferredAI/cornac). Multimodal baseline adapters reference the [MMRec](https://github.com/enoche/MMRec) and [MENTOR](https://github.com/Jinfeng-Xu/MENTOR) projects. The independent MACR-MF adapter follows the equations of [MACR](https://github.com/weitianxin/MACR) with the shape correction and protocol differences described in the manuscript. Preserve upstream notices and their applicable licensing terms. This update does not relicense third-party code or datasets.
