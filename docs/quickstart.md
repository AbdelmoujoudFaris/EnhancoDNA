# Quick Start

## 1. Generate the demo dataset

```bash
python scripts/generate_demo_dataset.py
```

This writes two synthetic, clearly-labelled DEMO structures to
`data/examples/` (`demo_tf_dna_two_factor.pdb` and
`demo_tf_dna_single_factor.pdb`) and 40 tiny labelled voxel-grid training
samples to `data/processed/demo_training/`.

## 2. Inspect a structure

```bash
enhancoai structure --input data/examples/demo_tf_dna_two_factor.pdb
```

## 3. Detect contacts

```bash
enhancoai contacts --structure data/examples/demo_tf_dna_two_factor.pdb \
    --protein A --protein B --dna C --dna D
```

## 4. Launch the GUI

```bash
enhancoai gui
```

Load the demo structure in the **Structure** tab, then explore
**Interactions**, **Cooperativity**, **AI**, **Visualisation** and
**Hypothesis Testing**.

## 5. Train a (demo) model

```bash
python scripts/train.py --config configs/training.yaml \
    --dataset-dir data/processed/demo_training --experiment-name demo_run
```

This is a *synthetic* label (presence/absence of a second protein chain),
useful for exercising the pipeline end to end -- it is not a scientific
result. See `experiments/demo_run/` for the checkpoint, metrics and
training curves.

## 6. Predict and explain

```bash
python scripts/predict.py --model experiments/demo_run/checkpoint.pt \
    --input data/examples/demo_tf_dna_two_factor.pdb --protein A --dna C --dna D \
    --embedding-dim 256
```

(`--embedding-dim` must match `model.embedding_dim` in the config used for
training; `configs/training.yaml` uses 256.)

## 7. Generate a report

```bash
python scripts/generate_report.py --input data/examples/demo_tf_dna_two_factor.pdb \
    --protein A --protein B --dna C --dna D --output-dir experiments/report_demo
```

Produces `report.html` and `report.pdf` with figures under `figures/`.
