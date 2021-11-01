---
sidebar_position: 7
title: Custom Evaluation Metric
---


Adding a custom metric to be logged, and perhaps even monitored for early stopping, is easy with classy. There are 3 ways to go about it:
* Hooking inside *ClassyPLModule*
* Using classy *PredictionCallback*-s
* Using Pytorch Lightning *Callback*-s


## Hooking inside *ClassyPLModule*

:::caution

Note that this approach cannot be applied for metrics that require generation, such as BLEU and Rouge.

:::

As *ClassyPLModule* is a subclass of a *LightningModule*, you can use the standard Lightning patterns:

* add code to your *validation_step* hook (and *training_step* hook if you want to monitor your metric also on training):

```python
class MyCustomClassyPLModule(ClassyPLModule):

    def __init__(self, *args, **kwargs):
        ...
        # init your metric here
        # using accuracy as the example
        self.accuracy_metric = torchmetrics.Accuracy()
    
    def validation_step(self, batch: dict, batch_idx: int) -> None:
        ...
        # compute accuracy
        predictions = torch.argmax(forward_output.logits, dim=-1)
        self.accuracy_metric(predictions, batch["labels"].squeeze(-1))
        self.log("val_accuracy", self.accuracy_metric, prog_bar=True)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        ...
        # add code here to monitor your metric also on training
        pass

```

* using the *validation_epoch_end* hook
```python
class MyClassyPLModule(ClassyPLModule):

    def validation_step(self, batch, batch_idx):
        ...
        # compute and return your predictions, along with any other object you might need for your metric (here we use labels as example)
        return predictions, labels


    def validation_epoch_end(self, validation_step_outputs):
        # compute here your validation metric using val_step_outputs
        # validation_step_outputs is a list containing all objects returned by validation_step
        for batch_predictions, batch_labels in validation_step_outputs:
            ...
```


## Using classy *PredictionCallback*-s

Sometimes, your metric might involve invoking complex code (perhaps some external non-Python library you have to invoke via bash)
or using a *forward* method that differs from the standard loss-computing one. For instance, generation models have different behaviors 
between training and inference time:
* They use teacher-forcing at training time
* They use decoding strategy such as Beam Search at inference time

While theoretically you could do this within *validation_epoch_end*, your code is likely to become a mess in no time. Rather, classy supports
this in a simple way via *PredictionCallback*-s:

```python
class PredictionCallback:
    def __call__(
        self,
        name: str,  # special name that identify the current __call__ invocation (required as a PredictionCallback can be invoked multiple times at each validation epoch, more on this later)
        predicted_samples: List[  
            Tuple[
                Union[SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample], 
                Union[str, List[str], Tupl[int, int]
            ]
        ],
        model: ClassyPLModule,  # model being trained
        trainer: pl.Trainer,  # Pytorch Lightning Trainer object
    ):
        raise NotImplementedError
```

*predicted_samples* is the list of your samples (*predicted_sample[i][0]*), along with their predictions (*predicted_sample[i][1]*).

:::tip

If available, you can access the gold labels via *predicted_sample[i][0].get_current_classification()* or the sample-specific variables (*predicted_sample[i][0].labels* for TokensSample, *predicted_sample[i][0].target_sequence* for GenerationSample, ...).
:::

:::info

A *PredictionCallback* is a classy callback invoked by *PredictionPLCallback*, a Pytorch Lightning *Callback*. Simply put, PredictionPLCallback takes as input a
list of *PredictionCallback*-s and, at each validation epoch, uses the model being trained to perform the prediction operation, invoking each *PredictionCallback*
on the result.

:::

To make an example, this is how you would introduce BLEU evaluation within your training:

```python title="classy/pl_callbacks/prediction/SacreBleuGenerationCallback"
from datasets import load_metric


class SacreBleuGenerationCallback(PredictionCallback):
    def __init__(self):
        self.bleu = load_metric("sacrebleu")

    def __call__(
        self,
        name: str,
        predicted_samples: List[Tuple[GenerationSample, str]],
        model: ClassyPLModule,
        trainer: pl.Trainer,
    ):

        assert all(sample.target_sequence is not None for sample, _ in predicted_samples)  # sanity check 

        references = [sample.target_sequence for sample, _ in predicted_samples]
        predictions = [prediction for _, prediction in predicted_samples]

        results = self.bleu.compute(predictions=predictions, references=[[r] for r in references])
        score = results["score"]

        model.log(f"{name}_bleu", score, prog_bar=True, on_step=False, on_epoch=True)
```

As for the yaml:

```yaml title="configurations/callbacks/bleu.yaml"
callbacks:
  - _target_: "classy.pl_callbacks.prediction.PredictionPLCallback"
    prediction_confs:
      - name: "validation"
        path: <PATH ON WHICH TO PERFORM THE EVALUATION>
        token_batch_size: 800
        prediction_param_conf_path: "configurations/prediction-params/generation-beam.yaml"
        limit: 1000
        enabled_prediction_callbacks:
          - "sacrebleu"
    prediction_callbacks:
      sacrebleu:
        _target_: "classy.pl_callbacks.prediction.SacreBleuGenerationCallback"
    prediction_dataset_conf: ${prediction.dataset}
```

As you can see, *PredictionPLCallback* takes as input a list of *prediction_confs*, each with an identifying name, allowing you to specify multiple evaluation paths whose BLEU you want to monitor,
and/or multiple prediction params (for instance, different beam sizes).

Finally, start the training:

```bash
classy train ... -c callbacks=bleu
```

## Using Pytorch Lightning *Callback*s

*PredictionCallback* is meant to make introducing new metrics easier. However, should you find its interface to be limiting, you can just implement your own Pytorch Lightning *Callback*,
inserting your metric computation inside it:

```python title="classy/pl_callbacks/my_custom_callback.py"
class MyCustomPLCallback(pl.Callback):
    def __init__(self, *args, **kwargs):
        # init your callback
        pass

    def on_validation_epoch_start(self, trainer: pl.Trainer, model: ClassyPLModule) -> None:
        # perform your prediction
        predictions = ...
        #  perform your evaluiation
        metric_name = ...
        score = ...
        # log
        model.log(metric_name, score, prog_bar=True, on_step=False, on_epoch=True)
```

Then, write the .yaml configuration:

```yaml title="configurations/callbacks/my-custom-callback.yaml"
callbacks:
  - _target_: "classy.pl_callbacks.my_custom_callback.MyCustomPLCallback"
    # add here init params
```

Finally, start your training:

```bash
classy train ... -c callbacks=my-custom-callback
```