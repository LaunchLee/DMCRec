# Dataset inputs for the revised study

The current manuscript evaluates Amazon Baby and Sports. Obtain their
interaction files and supplied image/text features using the dataset links
maintained by [MMRec](https://github.com/enoche/MMRec). The authors’
[Google Drive dataset folder](https://drive.google.com/drive/folders/1jnsSNG8FH8zkYaGXiZ2pxzey2eeC_k5l?usp=sharing)
shares the datasets used in the earlier experiments. This is a dataset download
link, separate from the Python reproduction entry point and the revised result archive.
When reusing these inputs, check the required files, identity mappings and input
fingerprints against the revised protocol. Third-party dataset terms apply.

Extract the two archives with this layout:

```text
datasets/
  MMRec-Baby/
    baby.inter
    image_feat.npy
    text_feat.npy
    i_id_mapping.csv
    u_id_mapping.csv
  MMRec-Sports/
    sports.inter
    image_feat.npy
    text_feat.npy
    i_id_mapping.csv
    u_id_mapping.csv
```

From the repository root, pass `--data-dir datasets` explicitly:

```bash
python main.py --group Baby/baby_primary/DMCRec/full --data-dir datasets
```

This only previews the selected configuration. Add `--execute` to train,
and select appropriate hardware with `--device cuda` for full graph training.
Without `--data-dir`, the shared runner defaults to
`experiments/dmcrec/datasets`, which is a different location.

Preserve feature-row order and the supplied ID mappings. Revised experiments
use binary observed feedback and random 70/10/20 splits with split seed 42;
they do not use source `x_label` as split assignments. Dataset counts,
held-out counts and input fingerprints are included in
`research_results/data_protocol.json`. Original data, feature arrays and
large embedding caches are excluded from Git. Derived aggregate results
are versioned separately under `research_results/`.

The historical video-feature caches and Cornac-Clothing example are not
inputs for the revised Baby/Sports manuscript results.
