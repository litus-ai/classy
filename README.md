# classy

### Classy Commands

```bash
classy train (sequence | token | sentence-pair) <dataset-path> [--model-name] [--exp-name] [--device] [--root] [[-c|--config] training.pl_trainer.val_check_interval=1.0 data.pl_module.batch_size=16]
classy predict interactive <model-path> [--device]
classy predict file <model-path> <file-path> [-o|--output-path] [--token-batch-size] [--device]
classy serve <model-path> [-p|--port] [--token-batch-size] [--device]
```

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

```bash
black -l 120 classy/
```

## TODOs

- logging

- pip package

- models
  - look for and train "una mazzettata" of models

- docs
  - comments
  - tutorials (no lines of code / few lines of code / I know what I am doing)
- gradio / streamlit
- test (haha!)
- pre-commit black (github actions?)
- Dockerfile
- bash screenshots
