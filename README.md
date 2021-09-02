# classy

### Training

```bash
classy train (sequence | token | sentence-pair) <dataset-path> [--model-name] [--exp-name]
classy predict interactive <model-path>
classy predict file <model-path> <file-path> [-o|--output-path]
classy serve <model-path> [-p|--port]
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

- Sentence Pair Classification
- logging

- pip package
  - command 

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
