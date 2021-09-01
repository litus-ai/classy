# classy

### Training

classy train <exp-name> (sequence | token | sentence-pair) <model-name> <dataset-path>

### Current Commands

```bash
PYTHONPATH=$(pwd) python classy/classy/scripts/model/train.py \
  exp_name=debug \
  device=cuda \
  --config-name sequence-bert
```

```bash
PYTHONPATH=$(pwd) python classy/classy/scripts/model/predict.py \
  experiments/debug/2021-09-01/10-59-28/checkpoints/epoch=00-val_loss=0.32.ckpt \
  -t
```