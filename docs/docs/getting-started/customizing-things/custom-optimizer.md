---
sidebar_position: 6
title: Custom Optimizer
---


# Optimizers

Classy comes with a set of well established predefined Optimizers that you can easily plug in your experiments. At the moment we support:


### Adam
One of the most famous Optimizer for Natural Language Processing applications, __virtually ubiquitous__.

- :hammer: [Implementation](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
- :page_facing_up: [Paper](https://arxiv.org/abs/1412.6980)

To use it, put the following yaml lines in your own profile or config.

```yaml
model:
  optim_conf:
    _target_: classy.optim.factories.AdamWithWarmupFactory
    lr: 3e-5
    warmup_steps: 5000
    total_steps: ${training.pl_trainer.max_steps}
    weight_decay: 0.01
    no_decay_params:
      - bias
      - LayerNorm.weight
```


### AdamW
Adam implementation with weight decay fix as stated in the original paper.

- :hammer: [Implementation](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.AdamW)
- :page_facing_up: [Paper](https://arxiv.org/abs/1711.05101)

To use it, put the following yaml lines in your own profile or config.

```yaml
model:
  optim_conf:
    _target_: classy.optim.factories.AdamWWithWarmupFactory
    lr: 3e-5
    warmup_steps: 5000
    total_steps: ${training.pl_trainer.max_steps}
    weight_decay: 0.01
    no_decay_params:
      - bias
      - LayerNorm.weight
```


### Adafactor
An Optimizer that you should use in order to __reduce the VRAM memory usage__. Performances are almost on par with AdamW

- :hammer: [Implementation](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#adafactor-pytorch)
- :page_facing_up: [Paper](https://arxiv.org/abs/1804.04235)

To use it, put the following yaml lines in your own profile or config.

```yaml
model:
  optim_conf:
    _target_: classy.optim.factories.AdamWWithWarmupFactory
    lr: 2e-5
    warmup_steps: 5000
    total_steps: ${training.pl_trainer.max_steps}
    weight_decay: 0.01
    no_decay_params:
      - bias
      - LayerNorm.weight
```


### RAdam
A more recent Optimizer that stabilizes training and let's you __skip the warmup phase__. You can replace AdamW with RAdam in almost every scenario.

- :hammer: [Implementation](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#adafactor-pytorch)
- :page_facing_up: [Paper](https://github.com/LiyuanLucasLiu/RAdam)

To use it, put the following yaml lines in your own profile or config.

```yaml
model:
  optim_conf:
    _target_: classy.optim.factories.RAdamFactory
    lr: 3e-5
    weight_decay: 0.01
    no_decay_params:
      - bias
      - LayerNorm.weight
```

## Custom Optimizers

If you want to implement your own Optimizer and Learning Rate Scheduler you can simply create a class that inherits from ```classy.optim.TorchFactory``` and implement the ```__call__``` method returning either the Optimizer or a dictionary containing the Optimizer and the Scheduler in the following way:

```python
class AdagradWithWarmup(TorchFactory):
    """
    Factory for Adagrad optimizer with warmup learning rate scheduler
    reference paper for Adagrad: https://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(
        self,
        lr: float,
        warmup_steps: int,
        total_steps: int,
        weight_decay: float,
        no_decay_params: Optional[List[str]],
    ):
        super().__init__(weight_decay, no_decay_params)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, module: torch.nn.Module):
        optimizer = Adagrad(
            module.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, self.warmup_steps, self.total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
```

This ```__call__``` method should return any of the possible return types from the [```configure_optimizers```](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers) method of pytorch_lightning. But if you don't have to do fancy stuff this piece of code is everything you'll need :).

Then, you can use your own Optimizer in your experiments by specifing it in your profile or config.

```yaml
model:
  optim_conf:
    _target_: my_repo.optimization.AdagradWithWarmup
    lr: 3e-5
    weight_decay: 0.01
    no_decay_params:
      - bias
      - LayerNorm.weight
```
