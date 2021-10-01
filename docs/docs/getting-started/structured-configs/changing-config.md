---
sidebar_position: 2
title: Changing Config
---

Most .yaml configs used in classy simply define the parameters that are passed at construction time to actual Python
objects. For instance, the following config:

```yaml title="configurations/model/token.yaml"
_target_: 'classy.pl_modules.hf.HFTokensPLModule'
transformer_model: ${transformer_model}
use_last_n_layers: 1
fine_tune: True
```

refers to the instantiation of the following object:

```python title="classy/pl_modules/hf.py"
class HFTokensPLModule(...):
    
    def __init__(
        self,
        transformer_model: str,
        use_last_n_layers: int,
        fine_tune: bool,
    ):
        ...
```

In particular, the *\_target\_* property defines the object to be instantiated by specifying the Python module path to
the corresponding object.

To actually instantiate configs, you can use Hydra *instantiate* method:

```python
import hydra
hydra.utils.instantiate(conf)  # conf is expected to be a DictConfig object (essentially a more powerful Python Dict loaded via OmegaConf); you don't need to care about this detail 
```

:::tip

By default, the instantiation process is recursive. You can change this behavior with the *\_recursive\_* paramater.

:::

## Changing values

With this in mind, you can see how easy it is to change things. For instance, if you want to change the fine-tuning behavior
as in the previous Section, you just need to change the .yaml file:

```yaml title="configurations/model/token.yaml"
_target_: 'classy.pl_modules.hf.HFTokensPLModule'
transformer_model: ${transformer_model}
use_last_n_layers: 1
fine_tune: False
```

and this will reflect automatically on your instantiated HFTokensPLModule.

:::tip

Having to enter the configuration folder, open the yaml file, edit the desired field and start the run every time
can be a bit of a nuisance, especially if you then have to revert your changes if you get worse results. 
To avoid this, for simple modifications, you can use the *-c* option of classy train, to provide Hydra CLI overrides on params. For
instance, to change the fine tuning strategy, ```classy train ... -c model.fine_tune=False```.

:::

## Writing a new config

Similarly, if you want to write a new config, perhaps specifying a new super-cool model, you just need to specify what to instantiate
(the *\_target\_* param) and how (the other params) to instantiate it:

```yaml title="configurations/model/model-new.yaml"
_target_: <python-module-path-to-object>
param1: value1
param2: value2
...
```

## Lazy Instantiation

:::caution

This section involves fairly complex concepts. Feel free to skip it if you are not planning on extensively
editing classy configs for the moment.

:::

Things usually are as simple as described above. However, actual instantiation sometimes requires resources that you 
cannot specify in the .yaml config. For instance, the actual configuration and code of HFTokensPLModule is the following:

```yaml title="configurations/model/token.yaml"
_target_: 'classy.pl_modules.hf.HFTokensPLModule'
transformer_model: ${transformer_model}
use_last_n_layers: 1
fine_tune: True
optim_conf:
  _target_: classy.optim.factories.RAdamWithDecayFactory
  lr: 1e-5
  weight_decay: 0.01
  no_decay_params:
    - bias
    - LayerNorm.weight
```

```python title="classy/pl_modules/hf.py"
class HFTokensPLModule(...):
    
    def __init__(
        self,
        transformer_model: str,
        use_last_n_layers: int,
        fine_tune: bool,
        vocabulary: Vocabulary,  # <--
        optim_conf: omegaconf.DictConfig,
    ):
        ...
```

with *\_\_init\_\_* having two additional *problematic* parameters:
* *vocabulary*
* *optim_conf*

On the one hand, *vocabulary* is not something you usually know beforehand, as it will be automatically built by classy itself. 
On the other hand, *optim_conf* is your optimizer configuration, whose instantiation involves instantiating a PyTorch optimizer;
however, this latter operation is impossible as PyTorch optimizers take model parameters as input to their constructors,
which you don't have yet as the model and its weights are exactly what your are trying to instantiate.

Yet, this code and configuration work, despite the missing *vocabulary* field and the circle dependency with *optim_conf*.
This is because, first, classy knows it should pass the built vocabulary along the other instantiation params, and manually adds it
when instantiating classy models.

:::tip

When facing instantiation issues similar to this *vocabulary/* thing, you can apply this pattern to deal with them,
manually passing the additional parameter via kwargs to *hydra.utils.instantiate(...)*.

:::

Second, classy disables recursive instantiation on models, meaning that *optim_conf* will not be instantiated automatically.
Rather, *\_\_init\_\_* will receive a DictConfig object and will have to take care itself of instantiating it.