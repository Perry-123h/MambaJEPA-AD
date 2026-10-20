# MambaJEPA AD revision reproducibility package

This package contains an independently reconstructed, auditable implementation for the BankSim revision. It is not the unavailable original author code. The selective state-space module is a Mamba-inspired stable diagonal recurrence evaluated causally over 12 past customer events; it does not use the official `mamba-ssm` kernels.

## Data

Download `bs140513_032310.csv` from the `ealaxi/banksim1` BankSim dataset:

https://www.kaggle.com/datasets/ealaxi/banksim1

The accepted file must have SHA-256:

`e37006f76d993bfaec3a02d717b4f0bdc1ebfa5d36449e91fb7c07e225278377`

The raw dataset is not redistributed in this package.

## Run

```text
python src/run_revision_experiments.py --data /path/to/bs140513_032310.csv --config configs/revision.json --out results
```

The run writes model files, training logs, compressed validation and test predictions, validation-selected thresholds and fusion weights, metrics, resource measurements, repeated-seed ablations, category-blind results, and customer-clustered bootstrap intervals. `run_metadata.json` is the main machine-readable audit record.

## Masking definition

The 35% probability applies independently to each semantic field group for each training example. A continuous field is replaced by zero after training-set standardization, which is the training mean. An age, gender, or category one-hot group is replaced by an all-zero vector, a state distinct from every valid one-hot level. The unmasked full attribute vector is passed to the exponential-moving-average target encoder. The target encoder starts as a copy of the online attribute encoder and is updated after every optimizer step as `target = 0.99 * target + 0.01 * online`.

## Validation discipline

Steps 0-107 are training, 108-143 are validation, and 144-179 are test. Standardization and vocabularies use training only. Neural checkpoints, classification thresholds, and late-fusion weights use validation only. Test labels are used only for final metrics and paired customer-clustered confidence intervals.
