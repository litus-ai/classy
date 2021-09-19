# classy

### Classy Commands

```bash
classy train (sequence | token | sentence-pair) <dataset-path> [--model-name] [--exp-name] [--device] [--root] [[-c|--config] training.pl_trainer.val_check_interval=1.0 data.pl_module.batch_size=16]
classy predict interactive <model-path> [--device]
classy predict file <model-path> <file-path> [-o|--output-path] [--token-batch-size] [--device]
classy serve <model-path> [-p|--port] [--token-batch-size] [--device]
```

## TODOs

### V0.1
- **luigi**: random -> np.random everywhere
- **luigi**: add inference time to demo
- **edoardo**: create different profiles to train the various tasks (try to build them based on the gpu memory size and report the training time on different gpus)
- **edoardo**: look for and train "una mazzettata" of models
- **niccolò**: classy download
- **edoardo**: extractive classification
- docs
  - docusaurus
  - comment extensively at least all classes and some important function
  - write readme
- **niccolò**: package install
- **luigi**: Dockerfile

### Later on
- num_workers can't be >1 right now
- pre-commit black (github actions?)
- training on colab notebooks
- logging
- testing
